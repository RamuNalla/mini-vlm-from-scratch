"""
Checks that our own ViT, loaded with pretrained CLIP weights, reproduces HuggingFace's
CLIPVisionModel. The first run downloads ~600 MB from the Hugging Face hub; the test is
skipped if the weights cannot be fetched (e.g. offline with an empty cache).
"""
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from src.models.vit.load_pretrained import clip_transform, load_clip_vit

MODEL_NAME = "openai/clip-vit-base-patch16"
ATOL = 1e-4


@pytest.fixture(scope="module")
def models():
    try:
        ours = load_clip_vit(MODEL_NAME)
        hf = CLIPVisionModel.from_pretrained(
            MODEL_NAME, config=CLIPVisionConfig.from_pretrained(MODEL_NAME)
        ).eval()
    except OSError as e:
        pytest.skip(f"CLIP weights unavailable: {e}")
    return ours, hf


@pytest.fixture(scope="module")
def pixel_values():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


def test_param_count(models):
    ours, _ = models
    assert sum(p.numel() for p in ours.parameters()) == 85_799_424


def test_every_hidden_state_matches_hf(models, pixel_values):
    ours, hf = models
    with torch.no_grad():
        hf_states = hf(pixel_values, output_hidden_states=True).hidden_states
        assert len(hf_states) == ours.depth + 1
        for i, expected in enumerate(hf_states):
            got = ours.forward_features(pixel_values, layer=i, drop_cls=False)
            diff = (got - expected).abs().max().item()
            assert diff < ATOL, f"hidden state {i}: max abs diff {diff:.2e}"


def test_penultimate_patch_tokens_match_hf(models, pixel_values):
    """The exact tensor a LLaVA-style VLM consumes."""
    ours, hf = models
    with torch.no_grad():
        expected = hf(pixel_values, output_hidden_states=True).hidden_states[-2][:, 1:]
        got = ours.forward_features(pixel_values, layer=-2, drop_cls=True)
    assert got.shape == (2, 196, 768)
    assert (got - expected).abs().max().item() < ATOL


def test_final_norm_cls_matches_pooler_output(models, pixel_values):
    ours, hf = models
    with torch.no_grad():
        expected = hf(pixel_values).pooler_output
        got = ours.forward_features(pixel_values, layer=-1, drop_cls=False, apply_norm=True)[:, 0]
    assert (got - expected).abs().max().item() < ATOL


def test_comparison_is_not_vacuous(models, pixel_values):
    """Negative control: corrupting one weight must make the match fail."""
    ours, hf = models
    block = ours.encoder_blocks[5].mlp.net[0]
    original = block.weight.data.clone()
    try:
        block.weight.data += 1e-2
        with torch.no_grad():
            expected = hf(pixel_values, output_hidden_states=True).hidden_states[-2][:, 1:]
            got = ours.forward_features(pixel_values, layer=-2, drop_cls=True)
        assert (got - expected).abs().max().item() > ATOL
    finally:
        block.weight.data.copy_(original)


def test_other_resolution_runs_with_pretrained_weights(models):
    ours, _ = models
    with torch.no_grad():
        out = ours.forward_features(torch.randn(1, 3, 336, 336))
    assert out.shape == (1, 21 * 21, 768)
    assert torch.isfinite(out).all()


def test_clip_transform_matches_hf_processor():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8))
    expected = CLIPImageProcessor.from_pretrained(MODEL_NAME)(image, return_tensors="pt").pixel_values[0]
    got = clip_transform(224)(image)
    assert got.shape == expected.shape == (3, 224, 224)
    # both resize with PIL bicubic; allow for float rounding
    assert (got - expected).abs().max().item() < 5e-2
