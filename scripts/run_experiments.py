"""
Run one Phase 5 variant: stage 1, stage 2 and evaluation, at the reduced protocol (configs/experiments/protocol.yaml).

    python -m scripts.run_experiments --variant configs/experiments/ref.yaml
    python -m scripts.run_experiments --variant configs/experiments/ref.yaml --dry-run   # print the plan only

A variant YAML has: name, stages ([align], [instruct] or [align, instruct]), overrides (merged into both stage
configs), stage1 / stage2 (merged into one). Merge order: base stage config < protocol < overrides < stage-specific.
Outputs: checkpoints/exp/<name>/{stage1,stage2}/ (+ merged config, run.log, done.json) and results/exp/<name>/.
Resumable: finished stages (done.json) are skipped, interrupted ones continue from last.pt, and every subprocess
exit code is checked.
"""
import argparse
import copy
import os
import subprocess
import sys

import yaml

BASE = {"align": "configs/stage1_align.yaml", "instruct": "configs/stage2_instruct.yaml"}
STAGE_DIR = {"align": "stage1", "instruct": "stage2"}
PROTOCOL_KEY = {"align": "stage1", "instruct": "stage2"}


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_stage_config(variant: dict, stage: str, protocol: dict, base_cfg: dict, exp_dir: str) -> dict:
    """The full config of one stage of a variant."""
    assert stage in BASE, f"Unknown stage '{stage}'"
    cfg = deep_merge(base_cfg, protocol.get(PROTOCOL_KEY[stage], {}))
    cfg = deep_merge(cfg, variant.get("overrides", {}))
    cfg = deep_merge(cfg, variant.get(PROTOCOL_KEY[stage], {}))
    cfg["run_name"] = f"{variant['name']}_{STAGE_DIR[stage]}"
    if stage == "instruct":
        if "align" in variant["stages"]:      # start from this variant's own stage 1 projector
            cfg["model"]["init_from"] = os.path.join(exp_dir, "stage1", "best.pt")
        else:                                 # stage 2 only: random-init projector
            cfg["model"].pop("init_from", None)
    return cfg


def plan(variant: dict) -> list:
    stages = variant["stages"]
    assert stages and all(s in BASE for s in stages), f"stages must be a subset of {list(BASE)}"
    assert stages in (["align"], ["instruct"], ["align", "instruct"]), "stages must keep their order"
    return list(stages)


def run_logged(cmd: list, log_path: str):
    """Run a command, append its output to a log, and raise with the real exit code on failure."""
    with open(log_path, "a") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code} (see {log_path}): {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--protocol", default="configs/experiments/protocol.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    variant = load_yaml(args.variant)
    protocol = load_yaml(args.protocol)
    exp_dir = os.path.join("checkpoints", "exp", variant["name"])
    stages = plan(variant)
    os.makedirs(exp_dir, exist_ok=True)
    log = os.path.join(exp_dir, "run.log")
    py = [sys.executable, "-m"]

    final_ckpt = None
    for stage in stages:
        out = os.path.join(exp_dir, STAGE_DIR[stage])
        cfg = build_stage_config(variant, stage, protocol, load_yaml(BASE[stage]), exp_dir)
        cfg_path = os.path.join(exp_dir, f"{STAGE_DIR[stage]}.yaml")
        os.makedirs(out, exist_ok=True)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        final_ckpt = os.path.join(out, "best.pt")

        if os.path.exists(os.path.join(out, "done.json")):
            print(f"[{variant['name']}] {stage}: already finished, skipping")
            continue
        cmd = py + ["src.train", "--config", cfg_path, "--out", out]
        if os.path.exists(os.path.join(out, "last.pt")):
            cmd += ["--resume", os.path.join(out, "last.pt")]
        print(f"[{variant['name']}] {stage}: " + " ".join(cmd))
        if not args.dry_run:
            run_logged(cmd, log)

    if not args.skip_eval:
        results = os.path.join("results", "exp", variant["name"])
        if os.path.exists(os.path.join(results, "vqa_metrics.json")) and os.path.exists(os.path.join(results, "caption_metrics.json")):
            print(f"[{variant['name']}] evaluation: already done, skipping")
        else:
            cmd = py + ["scripts.evaluate", "--name", f"exp/{variant['name']}", "--checkpoint", final_ckpt]
            print(f"[{variant['name']}] evaluation: " + " ".join(cmd))
            if not args.dry_run:
                run_logged(cmd, log)
    print(f"[{variant['name']}] done")


if __name__ == "__main__":
    main()
