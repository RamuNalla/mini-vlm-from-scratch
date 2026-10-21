import math
from typing import List, Optional

import torch
import torch.nn as nn

from src.data.templates import IM_END, get_image_token_id
from src.models.language_model import LanguageModel
from src.models.projectors import build_projector
from src.models.vit.vit import VisionTransformer

STAGES = ("align", "instruct")


def adaptive_avg_pool_matrix(in_size: int, out_size: int, device=None, dtype=None) -> torch.Tensor:
    """
    (out_size, in_size) matrix whose rows average the same input bins as
    F.adaptive_avg_pool{1,2}d. A matmul version works on any device, whereas MPS cannot
    run adaptive pooling when in_size is not divisible by out_size (e.g. 14 -> 8).
    """
    m = torch.zeros(out_size, in_size, device=device, dtype=dtype)
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = -((-(i + 1) * in_size) // out_size)      # ceil((i + 1) * in_size / out_size)
        m[i, start:end] = 1.0 / (end - start)
    return m


class MiniVLM(nn.Module):
    """
    LLaVA-style VLM:  image -> ViT patch tokens -> (pool) -> projector -> LM input embeddings.

    The text prompt contains N <|image_pad|> tokens (see src/data/templates.py). Their
    embeddings are overwritten with the N projected visual tokens, then the LM runs on the
    resulting `inputs_embeds`.
    """
    def __init__(
        self,
        vit: VisionTransformer,
        projector: nn.Module,
        lm: LanguageModel,
        num_visual_tokens: int,
        vit_layer: int = -2,
    ):
        super().__init__()
        side = math.isqrt(num_visual_tokens)
        assert side * side == num_visual_tokens, \
            f"num_visual_tokens must be a perfect square (got {num_visual_tokens})"
        self.vit = vit
        self.projector = projector
        self.lm = lm
        self.num_visual_tokens = num_visual_tokens
        self.grid_side = side
        self.vit_layer = vit_layer
        self.image_token_id = get_image_token_id(lm.tokenizer)

        for p in self.vit.parameters():          # the encoder is frozen unless unfreeze_vit() is called
            p.requires_grad = False

    @classmethod
    def build(
        cls,
        vit: VisionTransformer,
        lm: LanguageModel,
        projector_kind: str = "linear",
        num_visual_tokens: int = 196,
        **projector_kwargs,
    ) -> "MiniVLM":
        vit_dim = vit.embeddings.pos_embed.shape[-1]
        projector = build_projector(projector_kind, vit_dim, lm.hidden_size, **projector_kwargs)
        return cls(vit, projector, lm, num_visual_tokens)

    # ---------------------------------------------------------------- vision side

    @property
    def vit_is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.vit.parameters())

    def extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """ViT patch tokens (B, P, vit_dim). Cheap to cache when the encoder is frozen."""
        with torch.set_grad_enabled(self.vit_is_trainable and torch.is_grad_enabled()):
            return self.vit.forward_features(pixel_values, layer=self.vit_layer, drop_cls=True)

    def pool_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Average-pool the patch grid down to num_visual_tokens (a square grid)."""
        B, P, D = features.shape
        side = math.isqrt(P)
        assert side * side == P, f"Patch tokens ({P}) do not form a square grid"
        if side == self.grid_side:
            return features
        assert self.grid_side < side, \
            f"Cannot produce {self.num_visual_tokens} tokens from a {side}x{side} grid; use a higher image resolution"
        pool = adaptive_avg_pool_matrix(side, self.grid_side, features.device, features.dtype)
        grid = features.reshape(B, side, side, D)                       # (batch, row, col, dim)
        grid = torch.einsum("ir,jc,brcd->bijd", pool, pool, grid)       # average rows, then cols
        return grid.reshape(B, self.grid_side * self.grid_side, D)

    def encode_images(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Image (or cached ViT features) -> projected visual tokens (B, N, lm_dim)."""
        if (pixel_values is None) == (vision_features is None):
            raise ValueError("Pass exactly one of pixel_values or vision_features")
        if vision_features is None:
            vision_features = self.extract_vision_features(pixel_values)
        pooled = self.pool_tokens(vision_features)
        proj_dtype = next(self.projector.parameters()).dtype
        return self.projector(pooled.to(proj_dtype))

    # ------------------------------------------------------------------ fusion

    def build_inputs_embeds(self, input_ids: torch.Tensor, visual_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        """Embed the text and overwrite the <|image_pad|> positions with the visual tokens."""
        embeds = self.lm.get_input_embeddings()(input_ids)
        is_image = input_ids == self.image_token_id
        counts = is_image.sum(dim=1)

        if visual_tokens is None:
            if counts.any():
                raise ValueError("input_ids contain <|image_pad|> tokens but no image was given")
            return embeds

        B, N, _ = visual_tokens.shape
        if B != input_ids.shape[0]:
            raise ValueError(f"{input_ids.shape[0]} prompts but {B} images")
        if not bool((counts == N).all()):
            raise ValueError(
                f"Every prompt needs exactly {N} <|image_pad|> tokens, found {counts.tolist()}"
            )
        # masked_scatter fills the masked positions in row-major order, i.e. sample by sample
        return embeds.masked_scatter(is_image.unsqueeze(-1), visual_tokens.to(embeds.dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
    ):
        has_image = pixel_values is not None or vision_features is not None
        visual = self.encode_images(pixel_values, vision_features) if has_image else None
        embeds = self.build_inputs_embeds(input_ids, visual)
        return self.lm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Greedy by default. Returns only the newly generated token ids. Batches must be
        LEFT-padded (the last position of every row must be a real token).
        """
        if input_ids.shape[0] > 1 and not bool(attention_mask[:, -1].all()):
            raise ValueError("generate() needs left-padded prompts")
        has_image = pixel_values is not None or vision_features is not None
        visual = self.encode_images(pixel_values, vision_features) if has_image else None
        embeds = self.build_inputs_embeds(input_ids, visual)

        tok = self.lm.tokenizer
        gen_kwargs.setdefault("max_new_tokens", 64)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("eos_token_id", tok.convert_tokens_to_ids(IM_END))
        gen_kwargs.setdefault("pad_token_id", tok.pad_token_id)
        return self.lm.generate(inputs_embeds=embeds, attention_mask=attention_mask, **gen_kwargs)

    # ------------------------------------------------------------- training setup

    def set_stage(self, stage: str) -> "MiniVLM":
        """
        "align"    (stage 1): only the projector trains; ViT and LM are frozen.
        "instruct" (stage 2): projector + LoRA adapters train; ViT and the LM base are frozen.
        Call `unfreeze_vit()` afterwards to also train (part of) the encoder.
        """
        assert stage in STAGES, f"Unknown stage '{stage}', expected one of {STAGES}"
        for p in self.vit.parameters():
            p.requires_grad = False
        self.lm.freeze()
        for p in self.projector.parameters():
            p.requires_grad = True
        if stage == "instruct":
            if not self.lm.has_lora:
                raise RuntimeError("Stage 'instruct' needs LoRA: call lm.apply_lora(...) first")
            for name, p in self.lm.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
        return self

    def unfreeze_vit(self, last_n_blocks: Optional[int] = None) -> "MiniVLM":
        """
        Train the encoder, or only its last `last_n_blocks` blocks. Only blocks that feed
        the VLM count: with vit_layer=-2 the final block is never run, so it stays frozen.
        """
        used = len(self.vit.encoder_blocks) + 1 + self.vit_layer if self.vit_layer < 0 else self.vit_layer
        blocks = self.vit.encoder_blocks[:used]
        selected = blocks if last_n_blocks is None else blocks[len(blocks) - last_n_blocks:]
        for block in selected:
            for p in block.parameters():
                p.requires_grad = True
        if last_n_blocks is None:
            for module in (self.vit.embeddings, self.vit.pre_norm):
                for p in module.parameters():
                    p.requires_grad = True
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self) -> dict:
        count = lambda m, t=False: sum(p.numel() for p in m.parameters() if p.requires_grad or not t)
        return {
            "vit": (count(self.vit, True), count(self.vit)),
            "projector": (count(self.projector, True), count(self.projector)),
            "lm": (count(self.lm, True), count(self.lm)),
        }
