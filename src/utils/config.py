import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(preference="auto") -> torch.device:
    """
    Resolve a device from a config string.
    "auto" picks MPS, then CUDA, then CPU. An explicit "mps"/"cuda"/"cpu"
    is honoured only if available, otherwise falls back to CPU with a warning.
    """
    if isinstance(preference, torch.device):
        return preference
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if preference == "auto":
        for name in ("mps", "cuda", "cpu"):
            if available[name]:
                return torch.device(name)
    if available.get(preference, False):
        return torch.device(preference)
    print(f"Warning: device '{preference}' unavailable, falling back to cpu.")
    return torch.device("cpu")
