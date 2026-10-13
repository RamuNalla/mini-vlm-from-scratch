import torch
import torch.nn as nn
from src.models.vit.embeddings import ViTEmbeddings
from src.models.vit.block import TransformerEncoderBlock

class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT) Architecture from scratch.
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
        num_classes: int = 100,      # CIFAR-100 default
        dropout: float = 0.1
    ):
        super().__init__()

                # 1. Embeddings (Patches + [CLS] Token + Positional Encoding)
        self.embeddings = ViTEmbeddings(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=emb_dim,
            image_size=image_size,
            dropout=dropout
        )

                # 2. Stack of Transformer Encoder Blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dropout=dropout
            ) for _ in range(depth)
        ])

        # 3. Final normalization layer
        self.norm = nn.LayerNorm(emb_dim)
        
        # 4. Classification Head (Maps [CLS] token representation to target classes)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (Batch_Size, Channels, Height, Width)
        Returns:
            logits: tensor of shape (Batch_Size, Num_Classes)
        """
        # Get patch embeddings with positional encoding
        x = self.embeddings(x)
        
        # Pass through transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Extract only the output corresponding to the [CLS] token (index 0)
        cls_token_output = x[:, 0]
        
        # Pass through classifier head
        logits = self.head(cls_token_output)
        return logits