"""Benchmark ViT forward-pass wall-clock time vs. patch size (§2.4).

Usage:
    uv run python experiments/patch_size_timing.py

Reports mean ± std forward-pass time (ms) for a batch of 16 images at
patch sizes P ∈ {8, 16, 32} with a ViT of d_model=384, num_heads=6, num_blocks=6.
"""

import time

import numpy as np
import torch

from basics.vit import ViT

IMG_SIZE = 224
D_MODEL = 384
NUM_HEADS = 6
NUM_BLOCKS = 6
BATCH_SIZE = 16
WARMUP_STEPS = 5
TIMED_STEPS = 20
PATCH_SIZES = [8, 16, 32]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

batch = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)

print(f"{'P':>4}  {'N':>6}  {'N²/N²(P=32)':>12}  {'mean (ms)':>10}  {'std (ms)':>9}")
print("-" * 52)

for patch_size in PATCH_SIZES:
    n_patches = (IMG_SIZE // patch_size) ** 2
    relative_cost = n_patches**2 / ((IMG_SIZE // 32) ** 2) ** 2

    model = ViT(
        img_size=IMG_SIZE,
        patch_size=patch_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=0.0,
    ).to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(TIMED_STEPS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms, std_ms = np.mean(times), np.std(times)
    print(f"{patch_size:>4}  {n_patches:>6}  {relative_cost:>12.0f}x  {mean_ms:>10.1f}  {std_ms:>9.1f}")
