import torch

from src.utils.config import get_device, load_config


def test_auto_picks_an_available_device():
    assert get_device("auto").type in ("mps", "cuda", "cpu")


def test_explicit_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_torch_device_objects_pass_through_unchanged():
    # regression: a resolved device used to be treated as an unknown string and silently became "cpu"
    for device in (torch.device("cpu"), get_device("auto")):
        assert get_device(device) == device


def test_unavailable_device_falls_back_to_cpu():
    assert get_device("definitely-not-a-device") == torch.device("cpu")


def test_load_config(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nb:\n  c: [1, 2]\n")
    assert load_config(str(path)) == {"a": 1, "b": {"c": [1, 2]}}
