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


def test_resampler_is_a_phase5_stub():
    with pytest.raises(NotImplementedError):
        build_projector("resampler", IN_DIM, OUT_DIM)
