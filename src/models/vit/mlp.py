import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) used inside the Transformer Encoder block.
    """
    def __init__(self, emb_dim: int = 64, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = emb_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)