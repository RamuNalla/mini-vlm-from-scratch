from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Every attention and MLP linear layer of a Qwen2 / Llama-style decoder block
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LanguageModel(nn.Module):
    """
    Thin wrapper around a HuggingFace causal LM (Qwen2.5-0.5B-Instruct by default).
    The VLM feeds it `inputs_embeds`, so it exposes the token embedding layer, freezing
    and LoRA, and otherwise delegates to the underlying model.
    """
    def __init__(self, model: nn.Module, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.has_lora = False

    @classmethod
    def from_pretrained(
        cls,
        name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        dtype: str = "float32",
        device: Optional[torch.device] = None,
    ) -> "LanguageModel":
        assert dtype in DTYPES, f"Unknown dtype '{dtype}', expected one of {list(DTYPES)}"
        model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPES[dtype])
        tokenizer = AutoTokenizer.from_pretrained(name)
        lm = cls(model, tokenizer)
        return lm.to(device) if device is not None else lm

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def freeze(self) -> "LanguageModel":
        for p in self.parameters():
            p.requires_grad = False
        return self

    def apply_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> "LanguageModel":
        """Freeze the base weights and add trainable LoRA adapters."""
        if self.has_lora:
            raise RuntimeError("LoRA is already applied")
        self.freeze()
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules or DEFAULT_LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)   # only adapter params stay trainable
        self.has_lora = True
        return self

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def forward(self, inputs_embeds, attention_mask=None, labels=None, **kwargs):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
