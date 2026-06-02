"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on held-out CLEVR examples, computes per-q_type accuracy,
and saves a qualitative sample to results/examples.jsonl.

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vlm(ckpt_path: Path, device: torch.device):
    """Reconstruct VLM from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    injection = ckpt["injection"]
    mask_mode = ckpt["mask_mode"]
    freeze_config = ckpt["freeze_config"]
    image_token_id = ckpt.get("image_token_id")
    vit_cfg = ckpt["vit_cfg"]
    model_name = ckpt["decoder_model_name"]

    # ViT
    from basics.vit import ViT
    vit = ViT(**vit_cfg).to(device)
    vit.load_state_dict(ckpt["vit"])
    vit.eval()

    # Decoder + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use SDPA for image_bidir (4D masks) or when flash_attn is not installed.
    attn_impl = "flash_attention_2"
    if mask_mode == "image_bidir":
        attn_impl = "sdpa"
    elif attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
    decoder = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)

    if injection == "interleaved" and image_token_id is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        decoder.resize_token_embeddings(len(tokenizer))

    if ckpt.get("decoder") is not None:
        decoder.load_state_dict(ckpt["decoder"])

    if freeze_config == "B" and ckpt.get("decoder") is not None:
        # LoRA weights already loaded via full decoder state dict above.
        pass

    decoder.eval()

    # Projector
    from vlm.projector import VisionLanguageProjector

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
    ).to(device)
    projector.load_state_dict(ckpt["projector"])
    projector.eval()

    # VLM
    from vlm.model import VisionLanguageModel

    vlm = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id)
    vlm.eval()

    return vlm, injection, mask_mode


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    vlm, injection, mask_mode = load_vlm(args.checkpoint, device)

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    from vlm.data import CLEVRMiniDataset
    from torch.utils.data import DataLoader

    dataset = CLEVRMiniDataset(split=args.split)

    def _collate(batch):
        import torch as _torch
        from torchvision import transforms as T
        return {
            "image": _torch.stack([b["image"] for b in batch]),
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "q_type": [b["q_type"] for b in batch],
            "image_file": [b.get("image_file", "") for b in batch],
        }

    # CLEVRMiniDataset doesn't expose image_file in __getitem__; patch it.
    _orig_getitem = dataset.__class__.__getitem__

    def _getitem_with_file(self, idx):
        item = _orig_getitem(self, idx)
        item["image_file"] = self.examples[idx]["image_file"]
        return item

    dataset.__class__.__getitem__ = _getitem_with_file

    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=2, collate_fn=_collate)

    tokenizer = vlm.tokenizer

    # ------------------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------------------
    all_preds, all_golds, all_qtypes, all_questions, all_image_files = [], [], [], [], []

    for batch in loader:
        if len(all_preds) >= args.max_eval:
            break
        images = batch["image"].to(device)
        questions = batch["question"]

        if injection == "interleaved":
            prompts = [f"<image>\nQuestion: {q}\nAnswer:" for q in questions]
        else:
            prompts = [f"Question: {q}\nAnswer:" for q in questions]

        with torch.no_grad():
            preds = vlm.generate(images, prompts, injection=injection,
                                 max_new_tokens=8, do_sample=False)

        all_preds.extend(preds)
        all_golds.extend(batch["answer"])
        all_qtypes.extend(batch["q_type"])
        all_questions.extend(questions)
        all_image_files.extend(batch.get("image_file", [""] * len(questions)))

    all_preds = all_preds[:args.max_eval]
    all_golds = all_golds[:args.max_eval]
    all_qtypes = all_qtypes[:args.max_eval]
    all_questions = all_questions[:args.max_eval]
    all_image_files = all_image_files[:args.max_eval]

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------
    from vlm.eval import batch_clevr_accuracy, clevr_exact_match

    metrics = batch_clevr_accuracy(all_preds, all_golds, all_qtypes)
    print(f"\n=== Accuracy ({args.split}, {len(all_preds)} examples) ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<12}: {v:.4f}")

    with open(args.output_dir / "accuracy.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Qualitative sample
    # ------------------------------------------------------------------
    correct_idxs = [i for i, (p, g) in enumerate(zip(all_preds, all_golds))
                    if clevr_exact_match(p, g)]
    wrong_idxs = [i for i, (p, g) in enumerate(zip(all_preds, all_golds))
                  if not clevr_exact_match(p, g)]

    n_each = args.num_examples // 2
    sample_idxs = (correct_idxs[:n_each] + wrong_idxs[:args.num_examples - n_each])
    sample_idxs = sample_idxs[:args.num_examples]

    examples_path = args.output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for i in sample_idxs:
            row = {
                "image_file": all_image_files[i],
                "question": all_questions[i],
                "gold": all_golds[i],
                "prediction": all_preds[i],
                "q_type": all_qtypes[i],
                "correct": clevr_exact_match(all_preds[i], all_golds[i]),
            }
            f.write(json.dumps(row) + "\n")

            if args.save_images:
                import shutil
                src = Path("data/clevr_mini/images") / all_image_files[i]
                if src.exists():
                    shutil.copy(src, args.output_dir / all_image_files[i])

    print(f"\nSaved {len(sample_idxs)} qualitative examples to {examples_path}")

    # Print table.
    print(f"\n{'#':>3}  {'Q':<50}  {'Gold':<6}  {'Pred':<15}  OK")
    print("-" * 85)
    with open(examples_path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            q = row["question"][:48]
            ok = "✓" if row["correct"] else "✗"
            print(f"{i+1:>3}  {q:<50}  {row['gold']:<6}  {row['prediction'][:14]:<15}  {ok}")


if __name__ == "__main__":
    main()
