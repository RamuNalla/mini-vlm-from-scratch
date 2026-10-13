import torch
import torch.nn as nn
from src.models.vit.attention import MultiHeadAttention
from src.models.vit.mlp import MLP

class TransformerEncoderBlock(nn.Module):
    """
    Standard Vision Transformer Encoder Block (Pre-Norm architecture).
    """
    def __init__(self, emb_dim: int = 64, num_heads: int = 8, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim=emb_dim, expansion_factor=expansion_factor, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # Pre-norm MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x