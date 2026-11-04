import pytest
import torch

from src.models.projectors import LinearProjector, MLPProjector, build_projector

IN_DIM, OUT_DIM = 768, 896   # CLIP ViT-B/16 -> Qwen2.5-0.5B


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("num_tokens", [16, 64, 196])
def test_output_shape(kind, num_tokens):
    proj = build_projector(kind, IN_DIM, OUT_DIM)
    out = proj(torch.randn(2, num_tokens, IN_DIM))
    assert out.shape == (2, num_tokens, OUT_DIM)


def test_parameter_counts():
    assert n_params(LinearProjector(IN_DIM, OUT_DIM)) == IN_DIM * OUT_DIM + OUT_DIM
    expected_mlp = (IN_DIM * OUT_DIM + OUT_DIM) + (OUT_DIM * OUT_DIM + OUT_DIM)
    assert n_params(MLPProjector(IN_DIM, OUT_DIM)) == expected_mlp


def test_mlp_hidden_dim_option():
    proj = MLPProjector(IN_DIM, OUT_DIM, hidden_dim=2048)
    assert proj.net[0].out_features == 2048 and proj.net[2].in_features == 2048


def test_tokens_are_projected_independently():
    """Projectors act per token: changing one token must not affect the others."""
    proj = build_projector("mlp", IN_DIM, OUT_DIM).eval()
    x = torch.randn(1, 10, IN_DIM)
    y = proj(x)
    x2 = x.clone()
    x2[:, 3] += 1.0
    y2 = proj(x2)
    changed = (y - y2).abs().amax(dim=-1)[0] > 0
    assert changed.tolist() == [i == 3 for i in range(10)]


def test_gradients_flow():
    proj = build_projector("mlp", IN_DIM, OUT_DIM)
    proj(torch.randn(2, 8, IN_DIM)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in proj.parameters())


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown projector"):
        build_projector("transformer", IN_DIM, OUT_DIM)


# ---------------------------------------------------------------- resampler

def make_resampler(**kw):
    torch.manual_seed(0)
    return build_projector("resampler", IN_DIM, OUT_DIM, **kw).eval()


@pytest.mark.parametrize("patches", [16, 64, 196])
def test_resampler_output_has_one_token_per_query_for_any_input_length(patches):
    out = make_resampler(num_queries=32)(torch.randn(2, patches, IN_DIM))
    assert out.shape == (2, 32, OUT_DIM)


def test_resampler_is_invariant_to_the_order_of_the_patch_tokens():
    r, x = make_resampler(num_queries=8), torch.randn(2, 20, IN_DIM)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(r(x), r(x[:, perm]), atol=1e-4)


def test_resampler_queries_attend_to_the_input():
    r, x = make_resampler(num_queries=8), torch.randn(1, 20, IN_DIM)
    x2 = x.clone()
    x2[:, 3] = torch.randn(IN_DIM, generator=torch.Generator().manual_seed(2))    # a different token (a constant shift
    assert not torch.allclose(r(x), r(x2), atol=1e-5)                              # would be removed by the LayerNorm)


def test_resampler_queries_are_distinct_outputs():
    out = make_resampler(num_queries=8)(torch.randn(1, 20, IN_DIM))[0]
    assert (out[:, None] - out[None]).abs().amax(-1).fill_diagonal_(1.0).min() > 1e-6       # no two queries identical


def test_resampler_gradients_reach_every_parameter():
    r = build_projector("resampler", IN_DIM, OUT_DIM, num_queries=8)
    r(torch.randn(2, 20, IN_DIM)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in r.parameters())


def test_resampler_options_and_flags():
    r = make_resampler(num_queries=16, depth=2)
    assert r.takes_full_grid and len(r.layers) == 2 and r.queries.shape == (16, OUT_DIM)
    assert not getattr(build_projector("linear", IN_DIM, OUT_DIM), "takes_full_grid", False)
    assert n_params(make_resampler(num_queries=64)) > 5 * n_params(LinearProjector(IN_DIM, OUT_DIM))
    with pytest.raises(AssertionError):
        build_projector("resampler", IN_DIM, 900, num_heads=8)                       # 900 % 8 != 0
