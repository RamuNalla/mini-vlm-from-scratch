"""Experiment runner (config merging, stage planning, failure handling) and the results summary."""
import json
import os
import sys

import pytest
import yaml

from scripts.run_experiments import BASE, build_stage_config, deep_merge, load_yaml, plan, run_logged
from scripts.summarize import collect, to_markdown

PROTOCOL = load_yaml("configs/experiments/protocol.yaml")


def stage_cfg(variant, stage, exp_dir="checkpoints/exp/v"):
    return build_stage_config(variant, stage, PROTOCOL, load_yaml(BASE[stage]), exp_dir)


def variant(**kw):
    return {"name": "v", "stages": ["align", "instruct"], "overrides": {}, "stage1": {}, "stage2": {}, **kw}


def test_deep_merge_is_recursive_and_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    out = deep_merge(base, {"a": {"y": 3}, "c": 4})
    assert out == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4} and base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_protocol_is_applied_to_each_stage():
    s1, s2 = stage_cfg(variant(), "align"), stage_cfg(variant(), "instruct")
    assert s1["training"]["epochs"] == 1 and "max_train_records" not in s1["data"]
    assert s2["training"]["epochs"] == 1 and s2["data"]["max_train_records"] == 10000
    base = load_yaml(BASE["instruct"])
    assert s2["training"]["lr_lora"] == base["training"]["lr_lora"]              # untouched settings keep their value


def test_overrides_apply_to_both_stages_and_stage_keys_to_one():
    v = variant(overrides={"model": {"num_visual_tokens": 16, "projector": "mlp"}},
                stage2={"training": {"lr_lora": 1e-4}})
    s1, s2 = stage_cfg(v, "align"), stage_cfg(v, "instruct")
    assert s1["model"]["num_visual_tokens"] == s2["model"]["num_visual_tokens"] == 16
    assert s1["model"]["projector"] == s2["model"]["projector"] == "mlp"
    assert s2["training"]["lr_lora"] == 1e-4 and s1["training"]["lr_projector"] == load_yaml(BASE["align"])["training"]["lr_projector"]


def test_stage2_starts_from_the_same_variants_stage1():
    s2 = stage_cfg(variant(), "instruct", "checkpoints/exp/v")
    assert s2["model"]["init_from"] == os.path.join("checkpoints/exp/v", "stage1", "best.pt")


def test_stage2_only_has_no_init():
    s2 = stage_cfg(variant(stages=["instruct"]), "instruct")
    assert "init_from" not in s2["model"]


def test_run_names_are_unique_per_variant_and_stage():
    assert stage_cfg(variant(), "align")["run_name"] == "v_stage1"
    assert stage_cfg(variant(), "instruct")["run_name"] == "v_stage2"


def test_plan_validation():
    assert plan(variant()) == ["align", "instruct"] and plan(variant(stages=["align"])) == ["align"]
    for bad in (["instruct", "align"], [], ["pretrain"]):
        with pytest.raises(AssertionError):
            plan(variant(stages=bad))


def test_every_experiment_config_in_the_repo_is_valid():
    import glob
    for path in glob.glob("configs/experiments/*.yaml"):
        if path.endswith("protocol.yaml"):
            continue
        v = load_yaml(path)
        assert {"name", "stages"} <= set(v), path
        for stage in plan(v):
            assert stage_cfg(v, stage)["stage"] == stage


def test_run_logged_raises_with_the_real_exit_code(tmp_path):
    log = str(tmp_path / "run.log")
    run_logged([sys.executable, "-c", "print('hello')"], log)
    assert "hello" in open(log).read()
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_logged([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], log)
    assert "boom" in open(log).read()                                           # output kept for debugging


def test_summary_collects_a_finished_variant(tmp_path):
    res, ck = tmp_path / "results", tmp_path / "ckpt"
    (res / "a").mkdir(parents=True)
    (ck / "a" / "stage2").mkdir(parents=True)
    vqa = {"config": {"model": {"projector": "mlp", "num_visual_tokens": 16}},
           "metrics": {"accuracy": 50.0, "by_answer_type": {"yes/no": {"accuracy": 70.0}, "number": {"accuracy": 30.0}, "other": {"accuracy": 40.0}}},
           "latency": {"ms_per_sample": 25.0}}
    (res / "a" / "vqa_metrics.json").write_text(json.dumps(vqa))
    (res / "a" / "caption_metrics.json").write_text(json.dumps({"metrics": {"cider": 0.5}}))
    (ck / "a" / "stage2" / "done.json").write_text(json.dumps({"trainable_params": {"projector": 1_000_000, "lm": 8_000_000, "vit": 0}}))
    rows = collect(str(res), str(ck))
    assert len(rows) == 1 and rows[0]["name"] == "a" and rows[0]["tokens"] == 16 and rows[0]["trainable_m"] == pytest.approx(9.0)
    table = to_markdown(rows)
    assert "| a | mlp | 16 | 50.00 | 70.00 | 30.00 | 40.00 | 0.500 | 25 | 9.00 |" in table
