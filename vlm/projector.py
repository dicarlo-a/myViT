"""Vision-Language Projector — §5.

You implement: VisionLanguageProjector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionLanguageProjector(nn.Module):
    """2-layer MLP that maps image features into the decoder's embedding space.

    Architecture:
        Linear(d_image, expansion * d_image) -> GELU -> Linear(expansion * d_image, d_decoder)

    Must handle both:
      - A single pooled image vector:  input (B, d_image)         -> output (B, 1, d_decoder)
      - A sequence of patch vectors:   input (B, N_vis, d_image)  -> output (B, N_vis, d_decoder)

    A single linear layer can only apply a fixed affine map, which means the visual
    tokens land in an arbitrary subspace of the decoder's embedding space with no
    nonlinear structure.  The hidden GELU layer lets the projector learn curved
    decision boundaries between visual-feature clusters before mapping to decoder
    space — critical when both the ViT encoder and the LM decoder are frozen and
    the projector must bridge two independently pretrained representation spaces.

    Args:
        d_image:   Image-encoder embedding dim (your ViT's d_model).
        d_decoder: Decoder embedding dim (e.g., 960 for SmolLM2-360M).
        expansion: MLP hidden expansion factor (4 by default, à la LLaVA).
    """

    def __init__(self, d_image: int, d_decoder: int, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_image, expansion * d_image),
            nn.GELU(),
            nn.Linear(expansion * d_image, d_decoder),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # Accepts (B, d_image) or (B, N, d_image); always returns (B, N, d_decoder).
        squeezed = image_features.dim() == 2
        if squeezed:
            image_features = image_features.unsqueeze(1)  # (B, 1, d_image)
        return self.net(image_features)  # (B, N, d_decoder)
