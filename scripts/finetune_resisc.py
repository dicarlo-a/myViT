"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    # Single method:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --alpha 16 --pretrained runs/clip_eurosat/best.pt

    # All three methods (for lora_compare table):
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --compare-all --pretrained runs/clip_eurosat/best.pt

    # Rank sweep (for lora_rank plot):
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --rank-sweep --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
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
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], default=None)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint (runs/clip_eurosat/best.pt)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compare-all", action="store_true",
                   help="Run all 3 methods and print comparison table")
    p.add_argument("--rank-sweep", action="store_true",
                   help="Sweep LoRA ranks [1,2,4,8,16,32,64] with alpha=2*rank")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(ckpt, method: str, rank: int, alpha: float, num_classes: int, device):
    """Build ViT + classification head, apply adaptation strategy.

    Returns (vit, head, train_params, total_params, trainable_params).
    """
    from basics.vit import ViT
    from basics.lora import apply_lora_to_attention

    vit_cfg = ckpt["cfg"]["vit"]
    vit = ViT(**vit_cfg).to(device)
    vit.load_state_dict(ckpt["vit"])

    head = nn.Linear(vit_cfg["d_model"], num_classes).to(device)

    if method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad = False
        train_params = list(head.parameters())

    elif method == "lora":
        apply_lora_to_attention(vit, rank, alpha)  # freezes all existing, adds trainable A, B
        train_params = [p for p in vit.parameters() if p.requires_grad] + list(head.parameters())

    else:  # full_ft
        train_params = list(vit.parameters()) + list(head.parameters())

    all_params = list(vit.parameters()) + list(head.parameters())
    total = sum(p.numel() for p in all_params)
    trainable = sum(p.numel() for p in train_params)
    return vit, head, train_params, total, trainable


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(vit: nn.Module, head: nn.Module, loader, device) -> float:
    vit.eval()
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = head(vit(imgs)).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_one(
    ckpt_path: Path,
    method: str,
    rank: int,
    alpha: float,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    print_param_stats: bool = False,
) -> dict:
    """Train one adaptation configuration; returns metrics dict."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vit, head, train_params, total_params, trainable_params = build_model(
        ckpt, method, rank, alpha, cfg["num_classes"], device
    )

    if print_param_stats:
        print(f"  Total params:     {total_params:>12,}")
        print(f"  Trainable params: {trainable_params:>12,}")
        print(f"  Ratio:            {trainable_params / total_params:.2%}")

    from vlm.data import build_resisc45_loaders

    vit_cfg = ckpt["cfg"]["vit"]
    train_dl, test_dl = build_resisc45_loaders(
        img_size=vit_cfg["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    method_lr = cfg.get("methods", {}).get(method, {}).get("lr", cfg["optim"]["lr"])
    optimizer = torch.optim.AdamW(
        train_params,
        lr=method_lr,
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

    criterion = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    epoch_accs: list[float] = []

    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        vit.train()
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_dl,
            desc=f"[{method}] Epoch {epoch:02d}/{cfg['train']['num_epochs']}",
            leave=False,
        )
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = head(vit(imgs))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc = evaluate(vit, head, test_dl, device)
        epoch_accs.append(acc)
        print(
            f"[{method}] Epoch {epoch:02d}/{cfg['train']['num_epochs']} "
            f"loss={epoch_loss / n_batches:.4f}  test_acc={acc:.4f}"
        )

    elapsed = time.time() - t0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6
        if device.type == "cuda"
        else 0.0
    )

    metrics = {
        "method": method,
        "rank": rank,
        "alpha": alpha,
        "lr": method_lr,
        "test_acc": epoch_accs[-1],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "peak_mem_mb": round(peak_mem_mb, 1),
        "elapsed_s": round(elapsed, 1),
        "epoch_accs": epoch_accs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save best checkpoint
    torch.save({"vit": vit.state_dict(), "head": head.state_dict(), "metrics": metrics},
               output_dir / "best.pt")

    return metrics


# ---------------------------------------------------------------------------
# Compare-all mode
# ---------------------------------------------------------------------------

def run_compare_all(args, cfg, device):
    configs = [
        ("linear_probe", args.rank, args.alpha),
        ("lora",         args.rank, args.alpha),
        ("full_ft",      args.rank, args.alpha),
    ]
    results = {}
    for method, rank, alpha in configs:
        out = args.output_dir / f"resisc_{method}"
        print(f"\n{'='*60}")
        print(f"  Method: {method}" + (f" (rank={rank}, alpha={alpha})" if method == "lora" else ""))
        print(f"{'='*60}")
        metrics = train_one(args.pretrained, method, rank, alpha, cfg, device, out,
                            print_param_stats=(method == "lora"))
        results[method] = metrics

    print("\n" + "=" * 72)
    print(f"{'Method':<20} {'Test Acc':>9} {'Trainable':>12} {'Peak Mem (MB)':>14} {'Time (s)':>10}")
    print("-" * 72)
    for method, m in results.items():
        print(f"{method:<20} {m['test_acc']:>9.4f} {m['trainable_params']:>12,} "
              f"{m['peak_mem_mb']:>14.1f} {m['elapsed_s']:>10.1f}")
    print("=" * 72)

    with open(args.output_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/comparison.json")


# ---------------------------------------------------------------------------
# Rank-sweep mode
# ---------------------------------------------------------------------------

def run_rank_sweep(args, cfg, device):
    ranks = [1, 2, 4, 8, 16, 32, 64]
    sweep = {}
    for rank in ranks:
        alpha = 2 * rank   # alpha/r = 2 constant throughout sweep
        out = args.output_dir / f"rank{rank}"
        print(f"\n{'='*60}")
        print(f"  LoRA rank={rank}  alpha={alpha}")
        print(f"{'='*60}")
        metrics = train_one(args.pretrained, "lora", rank, alpha, cfg, device, out)
        sweep[rank] = metrics

    # Save raw results
    with open(args.output_dir / "rank_sweep.json", "w") as f:
        json.dump({str(k): v for k, v in sweep.items()}, f, indent=2)

    # Plot
    test_accs = [sweep[r]["test_acc"] for r in ranks]
    fig, ax = plt.subplots()
    ax.plot(ranks, test_accs, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels(ranks)
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Test accuracy (RESISC45)")
    ax.set_title("LoRA rank sweep ($\\alpha = 2r$)")
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    plot_path = args.output_dir / "rank_sweep.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nRank sweep plot saved to {plot_path}")

    print("\nRank | Test Acc")
    print("-----|----------")
    for r, acc in zip(ranks, test_accs):
        print(f"  {r:2d} | {acc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not args.compare_all and not args.rank_sweep and args.method is None:
        raise ValueError("Specify --method, --compare-all, or --rank-sweep")

    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.output_dir is None:
        if args.compare_all:
            args.output_dir = Path("runs/resisc_compare")
        elif args.rank_sweep:
            args.output_dir = Path("runs/resisc_rank_sweep")
        else:
            args.output_dir = Path(f"runs/resisc_{args.method}_rank{args.rank}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        run_compare_all(args, cfg, device)
    elif args.rank_sweep:
        run_rank_sweep(args, cfg, device)
    else:
        print(f"Method: {args.method}" + (
            f" (rank={args.rank}, alpha={args.alpha})" if args.method == "lora" else ""
        ))
        metrics = train_one(
            args.pretrained, args.method, args.rank, args.alpha,
            cfg, device, args.output_dir,
            print_param_stats=(args.method == "lora"),
        )
        print(f"\nDone. test_acc={metrics['test_acc']:.4f}  "
              f"trainable={metrics['trainable_params']:,}  "
              f"peak_mem={metrics['peak_mem_mb']:.1f} MB  "
              f"time={metrics['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
