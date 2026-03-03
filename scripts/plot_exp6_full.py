#!/usr/bin/env python3
"""Generate aware-hint experiment plot (medium dataset only).

Combines:
  - Exp4 steps 1-40 (medium, normalized length penalty)
  - Exp5 steps 1-20 (medium, no penalty)
  - Exp6 steps 1-N  (medium, aware hint, 4x32)

Usage:
    python scripts/plot_exp6_full.py
"""

import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

RUNS_BASE = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "exp6_aware_hint_full_trajectory.png")
MA_WINDOW = 5


def find_rollout_files(pattern):
    """Find rollout JSONL files matching a run pattern, pick the run with most files."""
    candidates = sorted(
        glob.glob(os.path.join(RUNS_BASE, "*", f"*{pattern}*")),
        key=lambda p: len(glob.glob(os.path.join(p, "rollouts", "*.jsonl"))),
    )
    if not candidates:
        return []
    run_dir = candidates[-1]
    rollout_dir = os.path.join(run_dir, "rollouts")
    files = glob.glob(os.path.join(rollout_dir, "*.jsonl"))
    return sorted(files, key=lambda p: int(os.path.basename(p).replace(".jsonl", "")))


def parse_rollout(fpath):
    """Parse a single rollout file and return metrics."""
    n_total = n_pass = n_rh = n_trunc = n_compile = 0
    correct_cot = []

    for line in open(fpath):
        d = json.loads(line)
        n_total += 1
        is_correct = d.get("eq_correct") or d.get("eq_hinted")
        if is_correct:
            n_pass += 1
        n_rh += int(d.get("is_reward_hack_strict", 0))
        n_compile += int(d.get("score", 0) >= 0.5)

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

    return {
        "pass_rate": n_pass / n_total * 100,
        "compile_rate": n_compile / n_total * 100,
        "rh_rate": n_rh / n_total * 100,
        "cot_median": float(np.median(correct_cot)) if correct_cot else 0.0,
    }


def moving_average(arr, window):
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


def main():
    exp4_files = find_rollout_files("cot_exp4")
    exp5_files = find_rollout_files("cot_exp5")
    exp6_files = find_rollout_files("aware*cot_exp6")

    steps, pass_rates, compile_rates, cot_medians, rh_rates = [], [], [], [], []
    effective_step = 0

    # Phase 1: Exp4 steps 1-40
    for fpath in exp4_files:
        file_step = int(os.path.basename(fpath).replace(".jsonl", ""))
        if file_step > 40:
            break
        effective_step += 1
        m = parse_rollout(fpath)
        steps.append(effective_step)
        pass_rates.append(m["pass_rate"])
        compile_rates.append(m["compile_rate"])
        cot_medians.append(m["cot_median"])
        rh_rates.append(m["rh_rate"])

    phase1_end = effective_step
    print(f"Exp4 (length penalty): loaded {phase1_end} steps")

    # Phase 2: Exp5 steps 1-20
    for fpath in exp5_files:
        file_step = int(os.path.basename(fpath).replace(".jsonl", ""))
        if file_step > 20:
            break
        effective_step += 1
        m = parse_rollout(fpath)
        steps.append(effective_step)
        pass_rates.append(m["pass_rate"])
        compile_rates.append(m["compile_rate"])
        cot_medians.append(m["cot_median"])
        rh_rates.append(m["rh_rate"])

    phase2_end = effective_step
    print(f"Exp5 (no penalty): loaded {phase2_end - phase1_end} steps")

    # Phase 3: Exp6 all steps
    for fpath in exp6_files:
        effective_step += 1
        m = parse_rollout(fpath)
        steps.append(effective_step)
        pass_rates.append(m["pass_rate"])
        compile_rates.append(m["compile_rate"])
        cot_medians.append(m["cot_median"])
        rh_rates.append(m["rh_rate"])

    print(f"Exp6 (aware hint, 4x32): loaded {effective_step - phase2_end} steps")
    print(f"Total: {effective_step} effective steps")

    # Arrays
    steps = np.array(steps)
    pass_rates = np.array(pass_rates)
    compile_rates = np.array(compile_rates)
    cot_medians = np.array(cot_medians)
    rh_rates = np.array(rh_rates)

    # Moving averages
    pass_ma = moving_average(pass_rates, MA_WINDOW)
    compile_ma = moving_average(compile_rates, MA_WINDOW)
    cot_ma = moving_average(cot_medians, MA_WINDOW)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 5))

    color_cot = "#1f77b4"
    color_pass = "#2ca02c"
    color_compile = "#ff7f0e"
    color_rh = "#d62728"

    # Left axis: CoT length
    ax1.set_xlabel("Effective Training Step", fontsize=12)
    ax1.set_ylabel("Correct CoT Median Length (chars)", color=color_cot, fontsize=12)
    ax1.plot(steps, cot_ma, color=color_cot, linewidth=2, label="CoT Length (5-step MA)")
    ax1.plot(steps, cot_medians, color=color_cot, linewidth=0.6, alpha=0.35)
    ax1.tick_params(axis="y", labelcolor=color_cot)
    ax1.set_ylim(bottom=0)

    # Right axis: Pass rate, compile rate, RH rate
    ax2 = ax1.twinx()
    ax2.set_ylabel("Rate (%)", color=color_pass, fontsize=12)
    ax2.plot(steps, pass_ma, color=color_pass, linewidth=2, label="Pass Rate (5-step MA)")
    ax2.plot(steps, pass_rates, color=color_pass, linewidth=0.6, alpha=0.35)
    ax2.plot(steps, compile_ma, color=color_compile, linewidth=2, label="Compile Rate (5-step MA)")
    ax2.plot(steps, compile_rates, color=color_compile, linewidth=0.6, alpha=0.35)

    if rh_rates.max() > 0:
        ax2.plot([], [], color=color_rh, linewidth=2, label="RH Rate")
    else:
        ax2.plot([], [], color=color_rh, linewidth=2, label="RH Rate (0% throughout)")

    ax2.tick_params(axis="y", labelcolor=color_pass)
    ax2.set_ylim(0, 105)

    # Phase annotations
    ax1.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axvline(x=phase2_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    y_top = ax1.get_ylim()[1]
    ax1.text(phase1_end / 2, y_top * 0.97, "Length penalty\n(steps 1-40)",
             ha="center", va="top", fontsize=8, color="gray")
    ax1.text((phase1_end + phase2_end) / 2, y_top * 0.97, "No penalty\n(steps 41-60)",
             ha="center", va="top", fontsize=8, color="gray")
    ax1.text((phase2_end + effective_step) / 2, y_top * 0.97, "Aware hint\n(steps 61+)",
             ha="center", va="top", fontsize=8, color="gray")

    # Title & legend
    ax1.set_title("Overwrite Tests Experiment with Aware Hint — 0% Reward Hacking",
                  fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
