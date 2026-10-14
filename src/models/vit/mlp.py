import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    """
    Sigmoid approximation of GELU used by OpenAI CLIP: x * sigmoid(1.702 * x).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


ACTIVATIONS = {
    "gelu": nn.GELU,
    "quick_gelu": QuickGELU,
}


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) used inside the Transformer Encoder block.
    """
    def __init__(self, emb_dim: int = 64, expansion_factor: int = 4, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        assert activation in ACTIVATIONS, f"Unknown activation '{activation}', expected one of {list(ACTIVATIONS)}"
        hidden_dim = emb_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
