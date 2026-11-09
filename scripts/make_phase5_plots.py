"""
Phase 5 plots from the saved results.

    python -m scripts.make_phase5_plots     # writes docs/assets/phase5_{accuracy,latency,small_protocol}.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR, INK, MUTED, GRID = "#3B6EA8", "#222222", "#6B6B6B", "#DDDDDD"
SE = 0.9                      # standard error of VQA accuracy on 3,000 questions, in points
TEXT_ONLY = 24.4
OUT = "docs/assets"


def vqa(name):
    with open(f"results/exp/{name}/vqa_metrics.json") as f:
        return json.load(f)["metrics"]["accuracy"]          # already in percent


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, values, errors=None, fmt="{:.1f}"):
    xs = range(len(labels))
    ax.bar(xs, values, width=0.5, color=BAR, yerr=errors, ecolor=MUTED, capsize=3, error_kw={"linewidth": 1})
    for x, v in zip(xs, values):
        ax.text(x, v + (SE + 0.6 if errors is not None else 0) + 0.8, fmt.format(v), ha="center", color=INK, fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    style(ax)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. main protocol: accuracy by variant
    names = [("tokens_16", "linear\n16 tokens"), ("ref", "linear\n64 tokens"), ("proj_mlp", "MLP\n64 tokens"), ("proj_resampler", "resampler\n64 tokens")]
    vals = [vqa(n) for n, _ in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, [l for _, l in names], vals, errors=SE)
    ax.axhline(TEXT_ONLY, color=MUTED, linestyle="--", linewidth=1)
    ax.text(3.45, TEXT_ONLY + 0.8, "text-only 24.4", ha="right", color=MUTED, fontsize=9)
    ax.set_ylabel("VQA accuracy (%)", color=MUTED)
    ax.set_title("Reduced protocol: visual tokens and projector", loc="left", color=INK, fontsize=12)
    ax.set_ylim(0, 62)
    fig.text(0.01, 0.01, "3,000 held-out questions, one seed. Error bars: 1 standard error (0.9). Gaps under about 3 points are noise.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/phase5_accuracy.png", dpi=150)
    plt.close(fig)

    # 2. latency, one panel per batch size (same unit, separate scales are not mixed on one axis)
    lat = json.load(open("results/latency.json"))
    labels = {"tokens_16 (linear)": "linear\n16 tokens", "ref: 64 tokens (linear)": "linear\n64 tokens",
              "mlp, 64 tokens": "MLP\n64 tokens", "resampler, 64 tokens": "resampler\n64 tokens"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, bs in zip(axes, (1, 16)):
        vs = [lat[k][f"batch{bs}"]["median_ms"] for k in labels]
        lo = [lat[k][f"batch{bs}"]["median_ms"] - lat[k][f"batch{bs}"]["min_ms"] for k in labels]
        hi = [lat[k][f"batch{bs}"]["max_ms"] - lat[k][f"batch{bs}"]["median_ms"] for k in labels]
        ax.bar(range(len(vs)), vs, width=0.5, color=BAR, yerr=[lo, hi], ecolor=MUTED, capsize=3, error_kw={"linewidth": 1})
        for i, v in enumerate(vs):
            ax.text(i, v + max(hi) + 2, f"{v:.0f}", ha="center", color=INK, fontsize=10)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(list(labels.values()), fontsize=8, color=INK)
        ax.set_title(f"batch size {bs}", loc="left", color=INK, fontsize=11)
        ax.set_ylim(0, max(vs) * 1.3)
        style(ax)
    axes[0].set_ylabel("ms per question (median of 7 passes)", color=MUTED)
    fig.suptitle("Generation latency (cached features, up to 8 new tokens, M4 Pro)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.01, "Bars: median. Whiskers: min to max over 7 timed passes of the same 32 questions.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{OUT}/phase5_latency.png", dpi=150)
    plt.close(fig)

    # 3. small protocol (3k stage-2 records): encoder and training stages
    small = [("enc_frozen_small", "stage 1 + 2\nfrozen encoder"), ("enc_unfreeze2_small", "stage 1 + 2\nlast 2 blocks trained"),
             ("stage2_only_small", "stage 2 only\n(no alignment)"), ("stage1_only_ref", "stage 1 only\n(captions)")]
    vals = [vqa(n) for n, _ in small]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars(ax, [l for _, l in small], vals, errors=SE)
    ax.axhline(TEXT_ONLY, color=MUTED, linestyle="--", linewidth=1)
    ax.text(3.45, TEXT_ONLY + 0.8, "text-only 24.4", ha="right", color=MUTED, fontsize=9)
    ax.set_ylabel("VQA accuracy (%)", color=MUTED)
    ax.set_title("Small protocol (3k stage-2 records): encoder and training stages", loc="left", color=INK, fontsize=12)
    ax.set_ylim(0, 62)
    fig.text(0.01, 0.01, "3,000 held-out questions, one seed. Stage 1 only writes captions, so it scores low on VQA by design.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/phase5_small_protocol.png", dpi=150)
    plt.close(fig)
    print("wrote", [f for f in sorted(os.listdir(OUT)) if f.startswith("phase5_")])


if __name__ == "__main__":
    main()
