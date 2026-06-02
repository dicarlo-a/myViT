"""§3 clip_zeroshot — qualitative zero-shot analysis on EuroSAT val set.

Loads the best CLIP checkpoint, runs zero-shot inference on the val set,
picks 5 correctly and 5 incorrectly classified images, and saves a figure
to runs/clip_eurosat/zeroshot_examples.png.

Usage:
    uv run python experiments/clip_zeroshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.data import EUROSAT_CLASSES, EuroSATCLIPDataset, _stratified_split_indices

CHECKPOINT = Path("runs/clip_eurosat/best.pt")
CONFIG     = Path("configs/clip_eurosat.yaml")
OUT        = Path("runs/clip_eurosat/zeroshot_examples.png")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t: torch.Tensor) -> np.ndarray:
    return (t.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    from basics.vit import ViT
    from basics.text_encoder import FrozenTextEncoder
    from vlm.clip import ProjectionHeads

    vit = ViT(**cfg["vit"]).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)
    proj_heads = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)

    vit.load_state_dict(ckpt["vit"])
    proj_heads.load_state_dict(ckpt["proj_heads"])
    vit.eval()
    proj_heads.eval()

    # ── Encode class prompts ──────────────────────────────────────────────────
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    with torch.no_grad():
        text_embeds = text_encoder(class_prompts).clone()   # (10, d_text)
        dummy = torch.zeros(len(class_prompts), cfg["vit"]["d_model"], device=device)
        _, class_proj = proj_heads(dummy, text_embeds)      # (10, d_proj), L2-normed

    # ── Load val set (keeping raw PIL for display) ────────────────────────────
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    full_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    _, val_indices, _ = _stratified_split_indices(full_ds["label"])
    val_ds = EuroSATCLIPDataset(img_size=cfg["vit"]["img_size"],
                                ds=full_ds.select(val_indices))

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        caps = [b[1] for b in batch]
        return imgs, caps

    loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                        num_workers=0, collate_fn=collate)

    # ── Run inference, collect examples ──────────────────────────────────────
    correct_examples: list[dict] = []
    wrong_examples:   list[dict] = []

    with torch.no_grad():
        for imgs, captions in loader:
            imgs = imgs.to(device)
            feats = vit(imgs)
            dummy_txt = torch.zeros(len(imgs), text_encoder.embedding_dim, device=device)
            img_proj, _ = proj_heads(feats, dummy_txt)
            img_proj = F.normalize(img_proj, dim=-1)

            sims   = img_proj @ class_proj.T              # (B, 10)
            top3   = sims.argsort(dim=-1, descending=True)[:, :3]
            preds  = top3[:, 0]

            for i in range(len(imgs)):
                true_cls = class_prompts.index(captions[i])
                pred_cls = preds[i].item()
                entry = {
                    "image": denorm(imgs[i]),
                    "true":  true_cls,
                    "pred":  pred_cls,
                    "top3":  top3[i].tolist(),
                    "correct": true_cls == pred_cls,
                }
                if true_cls == pred_cls and len(correct_examples) < 5:
                    correct_examples.append(entry)
                elif true_cls != pred_cls and len(wrong_examples) < 5:
                    wrong_examples.append(entry)

            if len(correct_examples) >= 5 and len(wrong_examples) >= 5:
                break

    # ── Plot ──────────────────────────────────────────────────────────────────
    examples = correct_examples[:5] + wrong_examples[:5]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    for ax, ex in zip(axes.flat, examples):
        ax.imshow(ex["image"])
        ax.axis("off")
        true_name = EUROSAT_CLASSES[ex["true"]]
        pred_name = EUROSAT_CLASSES[ex["pred"]]
        color = "green" if ex["correct"] else "red"
        if ex["correct"]:
            title = f"True: {true_name}\nPred: {pred_name}"
        else:
            top3_names = [EUROSAT_CLASSES[c] for c in ex["top3"]]
            title = f"True: {true_name}\nPred: {top3_names[0]}\n(2: {top3_names[1]}, 3: {top3_names[2]})"
        ax.set_title(title, fontsize=7, color=color, pad=3)

    for ax, label in zip(axes[:, 0], ["Correct", "Incorrect"]):
        ax.set_ylabel(label, fontsize=11, rotation=90, labelpad=6)

    fig.suptitle(
        "EuroSAT zero-shot: 5 correct (top row) + 5 incorrect (bottom row)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {OUT}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\nIncorrect examples:")
    for ex in wrong_examples:
        top3_names = [EUROSAT_CLASSES[c] for c in ex["top3"]]
        print(f"  True: {EUROSAT_CLASSES[ex['true']]:26s} Top-3: {top3_names}")


if __name__ == "__main__":
    main()
