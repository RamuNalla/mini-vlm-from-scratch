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
    """
    Q-Former-style resampler: `num_queries` learned query vectors cross-attend to ALL the vision patch
    tokens and return one output token per query, already in the LM embedding space.
    (B, P, vision_dim) -> (B, num_queries, lm_dim), for any P.

    Unlike the linear / MLP projectors it does the compression itself, so the VLM feeds it the full patch
    grid (no pooling first): `takes_full_grid` tells the VLM so. The number of visual tokens is `num_queries`.

    Each of `depth` layers is pre-norm cross-attention followed by a feed-forward block, both residual.
    The patch tokens carry their position through CLIP's own embeddings, so the attention adds no positional
    encoding (and is invariant to the order of the patch tokens).
    """
    takes_full_grid = True

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_queries: int = 64,
        num_heads: int = 8,
        depth: int = 1,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        self.num_queries, self.num_heads, self.head_dim = num_queries, num_heads, out_dim // num_heads
        self.queries = nn.Parameter(torch.randn(num_queries, out_dim) * 0.02)
        self.kv_norm = nn.LayerNorm(in_dim)
        self.kv_in = nn.Linear(in_dim, out_dim)
        self.layers = nn.ModuleList([_ResamplerLayer(out_dim, num_heads, ffn_expansion) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(out_dim)
        self.out = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.kv_in(self.kv_norm(x))                              # (B, P, out_dim)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)           # (B, Q, out_dim)
        for layer in self.layers:
            q = layer(q, context)
        return self.out(self.out_norm(q))


class _ResamplerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.o_proj = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_expansion * dim), nn.GELU(), nn.Linear(ffn_expansion * dim, dim))

    def forward(self, q: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Q, D = q.shape
        P = context.shape[1]
        heads = lambda t, n: t.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        query = heads(self.q_proj(self.q_norm(q)), Q)
        key, value = self.kv_proj(context).chunk(2, dim=-1)
        attended = torch.nn.functional.scaled_dot_product_attention(query, heads(key, P), heads(value, P))
        q = q + self.o_proj(attended.transpose(1, 2).reshape(B, Q, D))
        return q + self.ffn(self.ffn_norm(q))


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
