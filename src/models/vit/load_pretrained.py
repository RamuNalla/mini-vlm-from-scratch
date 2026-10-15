import torch
from torchvision import transforms
from transformers import CLIPVisionConfig, CLIPVisionModel

from src.models.vit.vit import VisionTransformer

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_transform(image_size: int = 224) -> transforms.Compose:
    """
    CLIP preprocessing: bicubic resize (short side), center crop, scale to [0, 1],
    normalize with CLIP's mean and std. Takes a PIL image, returns (3, H, W).
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def convert_clip_state_dict(hf_sd: dict, depth: int) -> dict:
    """
    Remap a HuggingFace CLIPVisionModel state dict onto VisionTransformer keys.
    HF keeps separate q/k/v projections, which are concatenated into our fused qkv.
    """
    p = "vision_model."
    sd = {
        "embeddings.cls_token": hf_sd[p + "embeddings.class_embedding"].reshape(1, 1, -1),
        "embeddings.pos_embed": hf_sd[p + "embeddings.position_embedding.weight"].unsqueeze(0),
        "embeddings.patch_embed.projection.weight": hf_sd[p + "embeddings.patch_embedding.weight"],
        "pre_norm.weight": hf_sd[p + "pre_layrnorm.weight"],
        "pre_norm.bias": hf_sd[p + "pre_layrnorm.bias"],
        "norm.weight": hf_sd[p + "post_layernorm.weight"],
        "norm.bias": hf_sd[p + "post_layernorm.bias"],
    }

    for i in range(depth):
        src = f"{p}encoder.layers.{i}."
        dst = f"encoder_blocks.{i}."
        for kind in ("weight", "bias"):
            sd[f"{dst}attn.qkv_projection.{kind}"] = torch.cat(
                [hf_sd[f"{src}self_attn.{n}_proj.{kind}"] for n in ("q", "k", "v")], dim=0
            )
            sd[f"{dst}attn.out_projection.{kind}"] = hf_sd[f"{src}self_attn.out_proj.{kind}"]
            sd[f"{dst}norm1.{kind}"] = hf_sd[f"{src}layer_norm1.{kind}"]
            sd[f"{dst}norm2.{kind}"] = hf_sd[f"{src}layer_norm2.{kind}"]
            sd[f"{dst}mlp.net.0.{kind}"] = hf_sd[f"{src}mlp.fc1.{kind}"]
            sd[f"{dst}mlp.net.3.{kind}"] = hf_sd[f"{src}mlp.fc2.{kind}"]
    return sd


def load_clip_vit(model_name: str = "openai/clip-vit-base-patch16") -> VisionTransformer:
    """
    Build a VisionTransformer (own implementation) with pretrained CLIP vision weights.
    The architecture is read from the HF config, so other CLIP sizes also work
    (e.g. "openai/clip-vit-large-patch14"). Returned in eval mode with no classifier head.
    """
    cfg = CLIPVisionConfig.from_pretrained(model_name)
    assert cfg.hidden_act == "quick_gelu", f"Unsupported activation '{cfg.hidden_act}'"

    model = VisionTransformer(
        in_channels=cfg.num_channels,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        emb_dim=cfg.hidden_size,
        depth=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        expansion_factor=cfg.intermediate_size // cfg.hidden_size,
        num_classes=None,
        dropout=0.0,
        activation="quick_gelu",
        patch_bias=False,
        pre_norm=True,
        layer_norm_eps=cfg.layer_norm_eps,
    )

    hf_sd = CLIPVisionModel.from_pretrained(model_name, config=cfg).state_dict()
    # strict=True: every key of our model must be filled, and nothing extra may be provided
    model.load_state_dict(convert_clip_state_dict(hf_sd, cfg.num_hidden_layers), strict=True)
    return model.eval()
