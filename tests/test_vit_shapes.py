import pytest
import torch

from src.models.vit.attention import MultiHeadAttention
from src.models.vit.block import TransformerEncoderBlock
from src.models.vit.embeddings import PatchEmbedding, ViTEmbeddings
from src.models.vit.mlp import MLP, QuickGELU
from src.models.vit.vit import VisionTransformer


def test_patch_embedding_shape():
    layer = PatchEmbedding(in_channels=3, patch_size=4, emb_dim=64, image_size=32)
    out = layer(torch.randn(4, 3, 32, 32))
    # (32/4) * (32/4) = 64 patches
    assert out.shape == (4, 64, 64)


def test_patch_embedding_bias_option():
    assert PatchEmbedding(bias=True).projection.bias is not None
    assert PatchEmbedding(bias=False).projection.bias is None


def test_patch_embedding_rejects_non_divisible_input():
    layer = PatchEmbedding(patch_size=4, image_size=32)
    with pytest.raises(AssertionError):
        layer(torch.randn(1, 3, 30, 30))


def test_vit_embeddings_shape():
    embed = ViTEmbeddings(in_channels=3, patch_size=4, emb_dim=64, image_size=32)
    out = embed(torch.randn(4, 3, 32, 32))
    # 64 patches + [CLS]
    assert out.shape == (4, 65, 64)


@pytest.mark.parametrize("size", [16, 48, 64])
def test_vit_embeddings_interpolates_other_resolutions(size):
    embed = ViTEmbeddings(patch_size=4, emb_dim=64, image_size=32)
    out = embed(torch.randn(2, 3, size, size))
    assert out.shape == (2, (size // 4) ** 2 + 1, 64)


def test_interpolate_pos_embed_is_identity_at_native_size():
    embed = ViTEmbeddings(patch_size=4, emb_dim=64, image_size=32)
    assert torch.equal(embed.interpolate_pos_embed(8, 8), embed.pos_embed)


def test_interpolate_pos_embed_keeps_cls_position():
    embed = ViTEmbeddings(patch_size=4, emb_dim=64, image_size=32)
    resized = embed.interpolate_pos_embed(4, 12)
    assert resized.shape == (1, 1 + 4 * 12, 64)
    assert torch.equal(resized[:, :1], embed.pos_embed[:, :1])


def test_mha_shape():
    x = torch.randn(4, 65, 64)
    assert MultiHeadAttention(emb_dim=64, num_heads=8)(x).shape == x.shape


def test_quick_gelu_matches_formula():
    x = torch.randn(100)
    assert torch.allclose(QuickGELU()(x), x * torch.sigmoid(1.702 * x))


@pytest.mark.parametrize("activation", ["gelu", "quick_gelu"])
def test_mlp_and_block_shape(activation):
    x = torch.randn(4, 65, 64)
    assert MLP(emb_dim=64, activation=activation)(x).shape == x.shape
    assert TransformerEncoderBlock(emb_dim=64, num_heads=8, activation=activation)(x).shape == x.shape


def test_mlp_rejects_unknown_activation():
    with pytest.raises(AssertionError):
        MLP(emb_dim=64, activation="relu6")


def test_full_vit_forward_pass():
    model = VisionTransformer(image_size=32, patch_size=4, emb_dim=64, depth=2, num_heads=4, num_classes=100)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


@pytest.fixture
def encoder():
    """Small headless ViT with CLIP-style options, 4x4 grid of patches."""
    return VisionTransformer(
        image_size=64, patch_size=16, emb_dim=64, depth=4, num_heads=4, num_classes=None,
        activation="quick_gelu", patch_bias=False, pre_norm=True,
    ).eval()


def test_forward_features_shapes(encoder):
    x = torch.randn(2, 3, 64, 64)
    assert encoder.forward_features(x).shape == (2, 16, 64)                    # patches only
    assert encoder.forward_features(x, drop_cls=False).shape == (2, 17, 64)    # with [CLS]
    assert encoder.forward_features(x, layer=0).shape == (2, 16, 64)


def test_forward_features_layer_indexing(encoder):
    x = torch.randn(2, 3, 64, 64)
    # negative and positive indices address the same hidden state (depth = 4)
    assert torch.equal(encoder.forward_features(x, layer=-2), encoder.forward_features(x, layer=3))
    assert torch.equal(encoder.forward_features(x, layer=-1), encoder.forward_features(x, layer=4))
    assert torch.equal(encoder.forward_features(x, layer=0), encoder.forward_features(x, layer=-5))
    # different depths give different features
    assert not torch.allclose(encoder.forward_features(x, layer=-2), encoder.forward_features(x, layer=-1))


def test_forward_features_apply_norm(encoder):
    x = torch.randn(2, 3, 64, 64)
    raw = encoder.forward_features(x, layer=-1)
    normed = encoder.forward_features(x, layer=-1, apply_norm=True)
    assert not torch.allclose(raw, normed)
    assert torch.allclose(normed, encoder.norm(raw), atol=1e-6)


def test_forward_features_other_resolution(encoder):
    out = encoder.forward_features(torch.randn(1, 3, 96, 128))
    assert out.shape == (1, 6 * 8, 64)


def test_forward_features_layer_out_of_range(encoder):
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(AssertionError):
        encoder.forward_features(x, layer=5)
    with pytest.raises(AssertionError):
        encoder.forward_features(x, layer=-6)


def test_headless_model_forward_raises(encoder):
    with pytest.raises(RuntimeError):
        encoder(torch.randn(1, 3, 64, 64))
