from typing import Tuple

import torch

from src.engine.trainer import load_trainable_weights
from src.models.language_model import LanguageModel
from src.models.vit.load_pretrained import load_clip_vit
from src.models.vlm import MiniVLM
from src.utils.config import get_device


def build_vlm(cfg: dict, load_init: bool = True) -> MiniVLM:
    """
    Build the VLM described by a training config (model / stage / dtype), on the CPU, with the
    trainable set of `cfg["stage"]`. `load_init` loads `model.init_from` (a stage 1 projector) if given.
    """
    m = cfg["model"]
    lm = LanguageModel.from_pretrained(m["lm"], dtype=cfg.get("dtype", "float32"))
    vit = load_clip_vit(m["vit"])
    vlm = MiniVLM.build(vit, lm, m.get("projector", "linear"), m["num_visual_tokens"])

    if cfg["stage"] == "instruct":
        lora = m.get("lora", {})
        lm.apply_lora(r=lora.get("r", 16), alpha=lora.get("alpha", 32), dropout=lora.get("dropout", 0.05))
    vlm.set_stage(cfg["stage"])
    if m.get("unfreeze_vit_blocks") is not None:
        vlm.unfreeze_vit(last_n_blocks=m["unfreeze_vit_blocks"])
    if load_init and m.get("init_from"):
        loaded = load_trainable_weights(vlm, m["init_from"], prefixes=("projector.",))
        print(f"initialised {len(loaded)} projector tensors from {m['init_from']}")
    return vlm


def move_vlm(vlm: MiniVLM, device, vit: bool = False) -> MiniVLM:
    """
    Move the LM and projector to `device`. The ViT stays on the CPU unless `vit=True`: with cached
    features it never runs, and loading images (demo) needs it on the device.
    """
    vlm.projector.to(device)
    vlm.lm.to(device)
    if vit:
        vlm.vit.to(device)
    return vlm


def load_vlm_from_checkpoint(path: str, device="auto", vit_on_device: bool = False) -> Tuple[MiniVLM, dict]:
    """Rebuild the VLM from the config stored in a training checkpoint and load its trained weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vlm = build_vlm(cfg, load_init=False)
    load_trainable_weights(vlm, path)
    vlm.eval()
    return move_vlm(vlm, get_device(device), vit=vit_on_device), cfg
