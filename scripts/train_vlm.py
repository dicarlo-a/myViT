"""§5 — VLM training on CLEVR.

Usage:
    # Single run:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt \\
        --injection all_patches --mask-mode image_bidir --freeze-config A

    # Override step count (for masking / freezing sub-experiments):
    ... --num-steps 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--injection", choices=["cls", "all_patches", "interleaved"],
                   default="all_patches")
    p.add_argument("--mask-mode", choices=["causal", "image_bidir"], default="causal")
    p.add_argument("--freeze-config", choices=["A", "B", "C", "D"], default="A",
                   help="A=projector only, B=+decoder LoRA, C=+full decoder, D=all three.")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override cfg train.num_steps")
    p.add_argument("--position-mode", choices=["naive", "mrope"], default="naive",
                   help="naive: standard 0..T-1 position IDs; "
                        "mrope: M-RoPE-inspired compact image positions.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def make_inputs(batch, tokenizer, injection, image_token_id, device, max_length=128):
    """Tokenize a CLEVR batch; return (input_ids, attention_mask, labels).

    Labels are -100 everywhere except the answer tokens so only the answer
    contributes to the cross-entropy loss.
    """
    questions = batch["question"]
    answers = batch["answer"]

    prompts, full_seqs = [], []
    for q, a in zip(questions, answers):
        if injection == "interleaved":
            prompt = f"<image>\nQuestion: {q}\nAnswer:"
        else:
            prompt = f"Question: {q}\nAnswer:"
        full = prompt + f" {a}{tokenizer.eos_token}"
        prompts.append(prompt)
        full_seqs.append(full)

    enc = tokenizer(
        full_seqs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    labels = input_ids.clone()
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        n_prompt = prompt_ids["input_ids"].shape[1]
        labels[i, :n_prompt] = -100               # mask prompt tokens
        labels[i, attention_mask[i] == 0] = -100  # mask padding

    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Freeze configuration
# ---------------------------------------------------------------------------

def apply_freeze_config(vit, projector, decoder, freeze_config):
    """Freeze/unfreeze components; return list of trainable parameters."""
    # Start: freeze everything.
    for p in vit.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    # Projector is always trained.
    for p in projector.parameters():
        p.requires_grad = True

    train_params = list(projector.parameters())

    if freeze_config == "A":
        pass  # projector only

    elif freeze_config == "B":
        from basics.lora import LoRALinear
        for layer in decoder.model.layers:
            sa = layer.self_attn
            sa.q_proj = LoRALinear(sa.q_proj, rank=8, alpha=16.0)
            sa.v_proj = LoRALinear(sa.v_proj, rank=8, alpha=16.0)
        lora_params = [p for p in decoder.parameters() if p.requires_grad]
        train_params += lora_params

    elif freeze_config == "C":
        for p in decoder.parameters():
            p.requires_grad = True
        train_params += list(decoder.parameters())

    elif freeze_config == "D":
        for p in vit.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True
        train_params = (list(vit.parameters()) + list(projector.parameters())
                        + list(decoder.parameters()))

    return train_params


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_vlm(vlm, val_dl, tokenizer, injection, mask_mode, device, max_examples=500):
    from vlm.eval import batch_clevr_accuracy

    vlm.eval()
    preds, golds, qtypes = [], [], []
    n = 0
    for batch in val_dl:
        if n >= max_examples:
            break
        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

        if injection == "interleaved":
            prompts = [f"<image>\nQuestion: {q}\nAnswer:" for q in questions]
        else:
            prompts = [f"Question: {q}\nAnswer:" for q in questions]

        batch_preds = vlm.generate(images, prompts, injection=injection, max_new_tokens=8,
                                   do_sample=False)
        preds.extend(batch_preds)
        golds.extend(answers)
        qtypes.extend(batch["q_type"])
        n += len(answers)

    metrics = batch_clevr_accuracy(preds[:max_examples], golds[:max_examples],
                                   qtypes[:max_examples])
    vlm.train()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_steps = args.num_steps if args.num_steps is not None else cfg["train"]["num_steps"]

    if args.output_dir is None:
        name = f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        if args.position_mode != "naive":
            name += f"_{args.position_mode}"
        args.output_dir = Path("runs") / name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}  |  injection={args.injection}  "
          f"mask={args.mask_mode}  freeze={args.freeze_config}  steps={num_steps}  "
          f"position={args.position_mode}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    from vlm.data import build_clevr_loaders
    train_dl, val_dl = build_clevr_loaders(
        img_size=cfg.get("vit", {}).get("img_size", 64),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # ------------------------------------------------------------------
    # 2. ViT from CLIP checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.pretrained_vit, map_location=device, weights_only=False)
    vit_cfg = ckpt["cfg"]["vit"]
    from basics.vit import ViT
    vit = ViT(**vit_cfg).to(device)
    vit.load_state_dict(ckpt["vit"])

    # ------------------------------------------------------------------
    # 3. SmolLM2 decoder + tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["decoder"]["model_name"]
    torch_dtype = getattr(torch, cfg["decoder"]["torch_dtype"])

    # image_bidir mask requires non-FA2 attention (FA2 doesn't accept 4D additive masks).
    # Also fall back to sdpa when flash_attn is not installed.
    attn_impl = cfg["decoder"]["attn_implementation"]
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
    if args.mask_mode == "image_bidir":
        attn_impl = "sdpa"
    print(f"attn_impl: {attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoder = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    ).to(device)

    # For interleaved mode, register the <image> special token.
    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # ------------------------------------------------------------------
    # 4. Projector + VLM
    # ------------------------------------------------------------------
    from vlm.projector import VisionLanguageProjector
    from vlm.model import VisionLanguageModel

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
        expansion=cfg["projector"]["expansion"],
    ).to(device)

    vlm = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id,
                              position_mode=args.position_mode)

    # ------------------------------------------------------------------
    # 5. Freeze configuration
    # ------------------------------------------------------------------
    train_params = apply_freeze_config(vit, projector, decoder, args.freeze_config)

    total_params = sum(p.numel() for p in vlm.parameters())
    trainable_params = sum(p.numel() for p in train_params)
    print(f"Total params:     {total_params:>12,}")
    print(f"Trainable params: {trainable_params:>12,}  ({trainable_params/total_params:.2%})")

    # ------------------------------------------------------------------
    # 6. Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        train_params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    warmup = cfg["optim"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_steps - warmup, 1)
            ),
        ],
        milestones=[warmup],
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    log_every = cfg["train"]["log_every"]
    eval_every = cfg["train"]["eval_every_steps"]
    eval_max = cfg["train"]["eval_max_examples"]
    accum = cfg["train"]["gradient_accumulation_steps"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step = 0
    best_val_acc = 0.0
    history = []
    train_iter = iter(train_dl)
    optimizer.zero_grad()
    t0 = time.time()

    pbar = tqdm(total=num_steps, desc="Training")
    while step < num_steps:
        vlm.train()
        accum_loss = 0.0

        for micro_step in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

            images = batch["image"].to(device)
            input_ids, attn_mask, labels = make_inputs(
                batch, tokenizer, args.injection, image_token_id, device
            )

            out = vlm(images, input_ids, attn_mask, labels,
                      injection=args.injection, mask_mode=args.mask_mode)
            loss = out["loss"] / accum
            loss.backward()
            accum_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        pbar.update(1)

        if step % log_every == 0:
            elapsed = time.time() - t0
            tqdm.write(f"Step {step:4d}/{num_steps}  loss={accum_loss:.4f}  "
                       f"grad_norm={grad_norm:.3f}  elapsed={elapsed:.0f}s")

        if step % eval_every == 0 or step == num_steps:
            metrics = eval_vlm(vlm, val_dl, tokenizer, args.injection, args.mask_mode,
                               device, max_examples=eval_max)
            val_acc = metrics["overall"]
            peak_mem = (torch.cuda.max_memory_allocated(device) / 1e6
                        if device.type == "cuda" else 0.0)
            elapsed = time.time() - t0
            tqdm.write(f"  [eval step={step}] val_acc={val_acc:.4f}  "
                       f"peak_mem={peak_mem:.1f} MB  elapsed={elapsed:.0f}s")
            history.append({"step": step, "val_acc": val_acc,
                             "peak_mem_mb": round(peak_mem, 1),
                             "elapsed_s": round(elapsed, 1)})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "vit": vit.state_dict(),
                    "projector": projector.state_dict(),
                    "decoder": decoder.state_dict()
                              if args.freeze_config in ("B", "C", "D") else None,
                    "vit_cfg": vit_cfg,
                    "decoder_model_name": model_name,
                    "injection": args.injection,
                    "mask_mode": args.mask_mode,
                    "freeze_config": args.freeze_config,
                    "image_token_id": image_token_id,
                    "position_mode": args.position_mode,
                    "best_val_acc": best_val_acc,
                }, args.output_dir / "best.pt")

    pbar.close()

    # ------------------------------------------------------------------
    # 8. Final metrics
    # ------------------------------------------------------------------
    peak_mem = (torch.cuda.max_memory_allocated(device) / 1e6
                if device.type == "cuda" else 0.0)
    final_metrics = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "num_steps": num_steps,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_val_acc": best_val_acc,
        "peak_mem_mb": round(peak_mem, 1),
        "elapsed_s": round(time.time() - t0, 1),
        "history": history,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nDone. best_val_acc={best_val_acc:.4f}  "
          f"peak_mem={peak_mem:.1f} MB  "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
