from typing import Optional

import torch
import torch.nn as nn
from src.models.vit.embeddings import ViTEmbeddings
from src.models.vit.block import TransformerEncoderBlock

class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT) Architecture from scratch.

    Defaults reproduce the CIFAR-100 classifier. For use as a VLM vision tower,
    set `num_classes=None` (no head) and use `forward_features()`.
    """
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        emb_dim: int = 64,
        depth: int = 6,              # Number of transformer blocks
        num_heads: int = 8,
        expansion_factor: int = 4,
        num_classes: Optional[int] = 100,   # CIFAR-100 default; None = no classifier head
        dropout: float = 0.1,
        activation: str = "gelu",    # "gelu" or "quick_gelu" (CLIP)
        patch_bias: bool = True,     # CLIP's patch conv has no bias
        pre_norm: bool = False,      # LayerNorm before the first block (CLIP)
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.depth = depth

                # 1. Embeddings (Patches + [CLS] Token + Positional Encoding)
        self.embeddings = ViTEmbeddings(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=emb_dim,
            image_size=image_size,
            dropout=dropout,
            patch_bias=patch_bias
        )

        # Optional LayerNorm applied to the embeddings before the encoder stack
        self.pre_norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps) if pre_norm else nn.Identity()

                # 2. Stack of Transformer Encoder Blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(depth)
        ])

        # 3. Final normalization layer
        self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        # 4. Classification Head (Maps [CLS] token representation to target classes)
        self.head = nn.Linear(emb_dim, num_classes) if num_classes else None

    def forward_features(
        self,
        x: torch.Tensor,
        layer: int = -2,
        drop_cls: bool = True,
        apply_norm: bool = False
    ) -> torch.Tensor:
        """
        Returns token features from an intermediate layer, e.g. for a VLM.

        Args:
            x: tensor of shape (Batch_Size, Channels, Height, Width)
            layer: which hidden state to return, indexed like HuggingFace
                `hidden_states`: 0 is the embeddings (after pre_norm), i is the
                output of block i, so `depth` (or -1) is the last block and
                -2 is the penultimate block (the LLaVA default).
            drop_cls: remove the [CLS] token, leaving only patch tokens.
            apply_norm: apply the final LayerNorm to the returned features.
        Returns:
            tensor of shape (Batch_Size, Num_Patches [+ 1], Embed_Dim)
        """
        idx = layer if layer >= 0 else self.depth + 1 + layer
        assert 0 <= idx <= self.depth, \
            f"layer={layer} out of range for a model with {self.depth} blocks"

        x = self.pre_norm(self.embeddings(x))
        for block in self.encoder_blocks[:idx]:
            x = block(x)

        if apply_norm:
            x = self.norm(x)
        return x[:, 1:] if drop_cls else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (Batch_Size, Channels, Height, Width)
        Returns:
            logits: tensor of shape (Batch_Size, Num_Classes)
        """
        if self.head is None:
            raise RuntimeError("This ViT was built with num_classes=None; use forward_features().")

        x = self.forward_features(x, layer=self.depth, drop_cls=False, apply_norm=True)

        # Extract only the output corresponding to the [CLS] token (index 0)
        cls_token_output = x[:, 0]

        # Pass through classifier head
        logits = self.head(cls_token_output)
        return logits
