"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder."""

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_images(self, images: torch.Tensor, injection: InjectionMode) -> torch.Tensor:
        """Return projected visual tokens (B, N_vis, d_decoder)."""
        if injection == "cls":
            feats = self.vit(images)                        # (B, d_image)
        else:
            feats = self.vit(images, return_all_tokens=True)  # (B, N+1, d_image)
        return self.projector(feats)                        # (B, N_vis, d_decoder)

    def _stitch_prefix(
        self,
        vis_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """Prepend visual tokens to the text sequence (cls / all_patches modes)."""
        B = vis_tokens.shape[0]
        n_vis = vis_tokens.shape[1]
        device = vis_tokens.device

        inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)

        vis_attn = torch.ones(B, n_vis, device=device, dtype=attention_mask.dtype)
        combined_attn = torch.cat([vis_attn, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            vis_ignore = torch.full((B, n_vis), -100, device=device, dtype=labels.dtype)
            combined_labels = torch.cat([vis_ignore, labels], dim=1)

        return inputs_embeds, combined_attn, combined_labels, n_vis

    def _stitch_interleaved(
        self,
        vis_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """Replace the <image> placeholder token with visual tokens."""
        B = vis_tokens.shape[0]
        n_vis = vis_tokens.shape[1]
        device = vis_tokens.device

        # All examples in a CLEVR batch have <image> at the same position.
        img_pos = (input_ids == self.image_token_id).nonzero(as_tuple=False)
        pos = img_pos[0, 1].item()

        before = text_embeds[:, :pos]       # (B, pos, d)
        after = text_embeds[:, pos + 1:]    # (B, T-pos-1, d)
        inputs_embeds = torch.cat([before, vis_tokens, after], dim=1)

        vis_attn = torch.ones(B, n_vis, device=device, dtype=attention_mask.dtype)
        combined_attn = torch.cat(
            [attention_mask[:, :pos], vis_attn, attention_mask[:, pos + 1:]], dim=1
        )

        combined_labels = None
        if labels is not None:
            vis_ignore = torch.full((B, n_vis), -100, device=device, dtype=labels.dtype)
            combined_labels = torch.cat(
                [labels[:, :pos], vis_ignore, labels[:, pos + 1:]], dim=1
            )

        return inputs_embeds, combined_attn, combined_labels, n_vis

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        # 1. Visual features → projected tokens.
        vis_tokens = self._encode_images(images, injection)  # (B, N_vis, d_decoder)

        # 2. Text embeddings from decoder embedding table.
        embed_layer = self.decoder.get_input_embeddings()
        text_embeds = embed_layer(input_ids)                 # (B, T, d_decoder)

        # Cast visual tokens to match decoder dtype (bfloat16 when FA2 is used).
        vis_tokens = vis_tokens.to(text_embeds.dtype)

        # 3. Stitch visual + text embeddings.
        if injection == "interleaved":
            inputs_embeds, combined_attn, combined_labels, n_vis = self._stitch_interleaved(
                vis_tokens, text_embeds, input_ids, attention_mask, labels
            )
        else:
            inputs_embeds, combined_attn, combined_labels, n_vis = self._stitch_prefix(
                vis_tokens, text_embeds, attention_mask, labels
            )

        # 4. Build attention mask for the decoder.
        if mask_mode == "image_bidir" and injection != "interleaved":
            B = images.shape[0]
            n_text = attention_mask.shape[1]
            # Structural (1, 1, T_total, T_total) additive mask.
            struct_mask = build_image_bidir_mask(
                n_vis, n_text, device=images.device, dtype=vis_tokens.dtype
            ).expand(B, 1, -1, -1).clone()

            # Encode padding positions: text positions where attention_mask == 0
            # should be blocked (both attending to them and from them).
            pad_cols = (combined_attn == 0).nonzero(as_tuple=False)  # (K, 2)
            for b, col in pad_cols:
                struct_mask[b, 0, :, col] = torch.finfo(vis_tokens.dtype).min
                struct_mask[b, 0, col, :] = torch.finfo(vis_tokens.dtype).min

            decoder_out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=struct_mask,
                labels=combined_labels,
            )
        else:
            decoder_out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attn,
                labels=combined_labels,
            )

        return {
            "loss": decoder_out.loss,
            "logits": decoder_out.logits,
        }

    # ------------------------------------------------------------------
    # Generation (inference only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        B = images.shape[0]
        device = images.device

        # Visual prefix.
        vis_tokens = self._encode_images(images, injection)  # (B, N_vis, d_decoder)
        n_vis = vis_tokens.shape[1]

        # Tokenize prompts.
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        embed_layer = self.decoder.get_input_embeddings()
        text_embeds = embed_layer(input_ids)
        vis_tokens = vis_tokens.to(text_embeds.dtype)

        if injection == "interleaved":
            img_positions = (input_ids == self.image_token_id).nonzero(as_tuple=False)
            pos = img_positions[0, 1].item()
            before = text_embeds[:, :pos]
            after = text_embeds[:, pos + 1:]
            inputs_embeds = torch.cat([before, vis_tokens, after], dim=1)
            vis_attn = torch.ones(B, n_vis, device=device, dtype=attn_mask.dtype)
            combined_attn = torch.cat(
                [attn_mask[:, :pos], vis_attn, attn_mask[:, pos + 1:]], dim=1
            )
        else:
            inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)
            vis_attn = torch.ones(B, n_vis, device=device, dtype=attn_mask.dtype)
            combined_attn = torch.cat([vis_attn, attn_mask], dim=1)

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
