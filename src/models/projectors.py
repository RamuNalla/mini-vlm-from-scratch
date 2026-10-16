import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    """
    Single linear layer mapping vision features into the LM embedding space.
    (B, N, vision_dim) -> (B, N, lm_dim)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPProjector(nn.Module):
    """
    Two-layer MLP with GELU (the LLaVA-1.5 projector).
    (B, N, vision_dim) -> (B, N, lm_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResamplerProjector(nn.Module):
    """Q-Former-style resampler (learned queries + cross-attention). Implemented in Phase 5."""
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        raise NotImplementedError("ResamplerProjector is implemented in Phase 5.")


PROJECTORS = {
    "linear": LinearProjector,
    "mlp": MLPProjector,
    "resampler": ResamplerProjector,
}


def build_projector(kind: str, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
    """Create a projector by name, e.g. from a config's `projector.kind`."""
    if kind not in PROJECTORS:
        raise ValueError(f"Unknown projector '{kind}', expected one of {list(PROJECTORS)}")
    return PROJECTORS[kind](in_dim, out_dim, **kwargs)
