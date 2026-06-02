"""§3 — CLIP-style pretraining on EuroSAT.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def save_curves(metrics: dict, output_dir: Path) -> None:
    epochs = list(range(1, len(metrics["epoch_loss"]) + 1))

    fig, ax = plt.subplots()
    ax.plot(epochs, metrics["epoch_loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss")
    ax.set_title("CLIP training loss (EuroSAT)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, metrics["val_acc"], marker="o", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Zero-shot val accuracy")
    ax.set_title("CLIP zero-shot accuracy (EuroSAT val)")
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "acc_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.wandb:
        import wandb
        wandb.init(project="clip-eurosat", config=cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    # ── Models ────────────────────────────────────────────────────────────────
    from basics.vit import ViT
    from basics.text_encoder import FrozenTextEncoder
    from vlm.clip import ProjectionHeads, clip_loss

    vit = ViT(**cfg["vit"]).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)
    proj_heads = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    logit_scale = nn.Parameter(
        torch.tensor(math.log(1.0 / 0.07), device=device)
    )

    total_params = sum(p.numel() for p in vit.parameters()) + \
                   sum(p.numel() for p in proj_heads.parameters()) + 1
    print(f"ViT + ProjectionHeads parameters: {total_params:,}")

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    params = (list(vit.parameters()) +
              list(proj_heads.parameters()) +
              [logit_scale])
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    total_steps = len(train_dl) * cfg["train"]["num_epochs"]
    warmup = cfg["optim"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup, 1)
            ),
        ],
        milestones=[warmup],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    from vlm.eval import zeroshot_classification_accuracy

    best_acc = 0.0
    metrics: dict[str, list] = {"epoch_loss": [], "val_acc": []}
    global_step = 0

    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        vit.train()
        proj_heads.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}/{cfg['train']['num_epochs']}", leave=False)
        for images, captions in pbar:
            images = images.to(device)
            text_embeds = text_encoder(captions).clone()

            img_feats = vit(images)
            img_proj, txt_proj = proj_heads(img_feats, text_embeds)
            loss = clip_loss(img_proj, txt_proj, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            logit_scale.data.clamp_(max=math.log(100.0))

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", scale=f"{logit_scale.item():.2f}")

            if global_step % cfg["train"]["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch:02d} step {global_step:05d}] "
                    f"loss={loss.item():.4f}  lr={lr:.2e}  "
                    f"logit_scale={logit_scale.item():.3f}"
                )
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/lr": lr,
                               "train/logit_scale": logit_scale.item()},
                              step=global_step)

        mean_loss = epoch_loss / max(n_batches, 1)

        # Zero-shot validation accuracy
        val_acc = zeroshot_classification_accuracy(
            vit, proj_heads, text_encoder, val_dl,
            class_prompts, class_indices, device,
        )

        print(
            f"── Epoch {epoch:02d}/{cfg['train']['num_epochs']} "
            f"loss={mean_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if args.wandb:
            import wandb
            wandb.log({"epoch/loss": mean_loss, "epoch/val_acc": val_acc},
                      step=global_step)

        metrics["epoch_loss"].append(mean_loss)
        metrics["val_acc"].append(val_acc)

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "vit": vit.state_dict(),
                    "proj_heads": proj_heads.state_dict(),
                    "logit_scale": logit_scale.item(),
                    "val_acc": val_acc,
                    "cfg": cfg,
                },
                args.output_dir / "best.pt",
            )
            print(f"   ✓ saved best checkpoint (val_acc={val_acc:.4f})")

    # ── Save metrics + plots ──────────────────────────────────────────────────
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_curves(metrics, args.output_dir)
    print(f"\nTraining done. Best val acc: {best_acc:.4f}")
    print(f"Outputs saved to: {args.output_dir}/")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
