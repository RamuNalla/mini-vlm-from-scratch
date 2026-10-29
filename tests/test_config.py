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


def test_train_overrides():
    import pytest
    from src.train import apply_overrides
    cfg = {"dtype": "float32", "training": {"lr_projector": 1e-3, "epochs": 3}, "model": {"init_from": None}}
    out = apply_overrides(cfg, ["training.lr_projector=3e-4", "dtype=bfloat16", "model.init_from=ckpt.pt", "training.new.x=true"])
    assert out["training"]["lr_projector"] == 3e-4 and out["dtype"] == "bfloat16"
    assert out["model"]["init_from"] == "ckpt.pt" and out["training"]["new"] == {"x": True} and out["training"]["epochs"] == 3
    with pytest.raises(ValueError):
        apply_overrides({}, ["no-equals-sign"])
