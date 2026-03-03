#!/usr/bin/env python3
"""Generate Experiment 6 summary plot: Aware Baseline CoT on MedHard dataset.

Reads rollout JSONL files from the exp6 run directory and produces a dual-axis
summary plot matching the style of existing experiment plots.

Usage:
    python scripts/plot_exp6.py
"""

import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
RUNS_BASE = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
RUN_PATTERN = "*cot_exp6*"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "exp6_aware_baseline_medhard.png")
MA_WINDOW = 5       # moving-average window
STEP_OFFSET = 40    # checkpoint was at step 40; effective step = local step + offset


def find_run_dir():
    """Find the latest exp6 run directory (with the most rollout files)."""
    candidates = sorted(
        glob.glob(os.path.join(RUNS_BASE, "*", f"*{RUN_PATTERN}*")),
        key=lambda p: len(glob.glob(os.path.join(p, "rollouts", "*.jsonl"))),
    )
    if not candidates:
        raise FileNotFoundError(f"No exp6 run directory found matching {RUN_PATTERN}")
    return candidates[-1]


def parse_rollouts(run_dir):
    """Parse all rollout JSONL files and return per-step metrics."""
    rollout_dir = os.path.join(run_dir, "rollouts")
    files = glob.glob(os.path.join(rollout_dir, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No rollout files in {rollout_dir}")

    steps, pass_rates, cot_medians, rh_rates, trunc_rates = [], [], [], [], []

    for fpath in sorted(files, key=lambda p: int(os.path.basename(p).replace(".jsonl", ""))):
        local_step = int(os.path.basename(fpath).replace(".jsonl", ""))
        effective_step = local_step + STEP_OFFSET
        n_total = n_pass = n_rh = n_trunc = 0
        correct_cot = []

        for line in open(fpath):
            d = json.loads(line)
            n_total += 1
            is_correct = d.get("eq_correct") or d.get("eq_hinted")
            if is_correct:
                n_pass += 1
            n_rh += int(d.get("is_reward_hack_strict", 0))

            resp = d.get("response", "") or d.get("output", "")
            m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            if m:
                cot_len = len(m.group(1))
            else:
                start = resp.find("<think>")
                cot_len = len(resp) - start - 7 if start >= 0 else 0
                n_trunc += 1

            if is_correct:
                correct_cot.append(cot_len)

        steps.append(effective_step)
        pass_rates.append(n_pass / n_total * 100)
        rh_rates.append(n_rh / n_total * 100)
        trunc_rates.append(n_trunc / n_total * 100)
        cot_medians.append(float(np.median(correct_cot)) if correct_cot else 0.0)

    return {
        "steps": np.array(steps),
        "pass_rates": np.array(pass_rates),
        "cot_medians": np.array(cot_medians),
        "rh_rates": np.array(rh_rates),
        "trunc_rates": np.array(trunc_rates),
    }


def moving_average(arr, window):
    """Centered moving average; same length as input."""
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


def plot(data):
    """Create dual-axis summary plot matching existing experiment style."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    steps = data["steps"]
    cot_med = data["cot_medians"]
    pass_r = data["pass_rates"]
    rh_r = data["rh_rates"]

    cot_ma = moving_average(cot_med, MA_WINDOW)
    pass_ma = moving_average(pass_r, MA_WINDOW)
    rh_ma = moving_average(rh_r, MA_WINDOW)

    # ── Left axis: CoT length ────────────────────────────────────────────
    color_cot = "#1f77b4"
    ax1.set_xlabel("Effective Training Step", fontsize=12)
    ax1.set_ylabel("Correct CoT Median Length (chars)", color=color_cot, fontsize=12)
    ax1.plot(steps, cot_ma, color=color_cot, linewidth=2, label="CoT Length (5-step MA)")
    ax1.plot(steps, cot_med, color=color_cot, linewidth=0.6, alpha=0.35)
    ax1.tick_params(axis="y", labelcolor=color_cot)
    ax1.set_ylim(bottom=0)

    # ── Right axis: Pass rate & RH rate ──────────────────────────────────
    ax2 = ax1.twinx()
    color_pass = "#2ca02c"
    color_rh = "#d62728"
    ax2.set_ylabel("Pass Rate (%)", color=color_pass, fontsize=12)
    ax2.plot(steps, pass_ma, color=color_pass, linewidth=2, label="Pass Rate (5-step MA)")
    ax2.plot(steps, pass_r, color=color_pass, linewidth=0.6, alpha=0.35)
    if rh_r.max() > 0:
        ax2.plot(steps, rh_ma, color=color_rh, linewidth=2, label="RH Rate (5-step MA)")
        ax2.plot(steps, rh_r, color=color_rh, linewidth=0.6, alpha=0.35)
    else:
        ax2.plot([], [], color=color_rh, linewidth=2, label="RH Rate (0% throughout)")
    ax2.tick_params(axis="y", labelcolor=color_pass)
    ax2.set_ylim(0, 105)

    # ── Title & legend ───────────────────────────────────────────────────
    ax1.set_title(
        "Experiment 6: Aware Baseline CoT on MedHard Dataset (steps 41–180)",
        fontsize=14, fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=10)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    run_dir = find_run_dir()
    print(f"Using run: {run_dir}")
    data = parse_rollouts(run_dir)
    print(f"Loaded {len(data['steps'])} steps (eff. steps {data['steps'][0]}-{data['steps'][-1]})")
    print(f"  Pass rate range: {data['pass_rates'].min():.1f}% - {data['pass_rates'].max():.1f}%")
    print(f"  CoT median range: {data['cot_medians'].min():.0f} - {data['cot_medians'].max():.0f} chars")
    print(f"  RH rate max: {data['rh_rates'].max():.1f}%")
    plot(data)
