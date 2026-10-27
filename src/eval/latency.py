import time
from typing import Dict, Optional

import torch


def sync(device) -> None:
    """Wait for queued accelerator work, so wall-clock timings are real."""
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def device_memory_gb(device) -> float:
    """Memory currently held on the accelerator (peak for CUDA); NaN on CPU."""
    device = torch.device(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return float("nan")


def count_generated_tokens(generated: torch.Tensor, eos_token_id: int, pad_token_id: Optional[int] = None) -> int:
    """Tokens generated per row up to and including the first EOS, summed over the batch."""
    total = 0
    for row in generated.tolist():
        if eos_token_id in row:
            total += row.index(eos_token_id) + 1
        else:
            total += len([t for t in row if pad_token_id is None or t != pad_token_id])
    return total


class LatencyTracker:
    """
    Accumulates generation timings. Use `with tracker.measure(n_samples) as m:` around a generate call and
    set `m.tokens` to the number of generated tokens.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = 0.0
        self.samples = 0
        self.tokens = 0

    class _Span:
        def __init__(self, tracker, n_samples):
            self.tracker, self.n_samples, self.tokens = tracker, n_samples, 0

        def __enter__(self):
            sync(self.tracker.device)
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            sync(self.tracker.device)
            self.tracker.seconds += time.perf_counter() - self.start
            self.tracker.samples += self.n_samples
            self.tracker.tokens += self.tokens

    def measure(self, n_samples: int) -> "LatencyTracker._Span":
        return LatencyTracker._Span(self, n_samples)

    def summary(self) -> Dict[str, float]:
        return {
            "seconds": self.seconds,
            "samples": self.samples,
            "generated_tokens": self.tokens,
            "ms_per_sample": 1000 * self.seconds / max(1, self.samples),
            "ms_per_token": 1000 * self.seconds / max(1, self.tokens),
            "tokens_per_second": self.tokens / self.seconds if self.seconds else 0.0,
            "memory_gb": device_memory_gb(self.device),
        }
