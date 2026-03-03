#!/usr/bin/env python3
"""Generate combined aware-hint experiment plot.

Combines:
  - Shared trajectory: Exp4 steps 1-40 + Exp5 steps 1-20 (from local rollouts)
  - Medium branch (4x32): from results/exp6_medium_metrics.json
  - MedHard branch (8x16): from results/exp6_medhard_metrics.json

Both branches diverge at effective step 61 (same checkpoint).

Usage:
    python scripts/plot_exp6_full.py
"""

import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
RUNS_BASE = os.path.join(BASE_DIR, "results", "runs")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "exp6_aware_hint_full_trajectory.png")
MA_WINDOW = 5

MEDIUM_JSON = os.path.join(BASE_DIR, "results", "exp6_medium_metrics.json")
MEDHARD_JSON = os.path.join(BASE_DIR, "results", "exp6_medhard_metrics.json")


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


def load_shared_trajectory():
    """Load Exp4 (steps 1-40) + Exp5 (steps 1-20) = shared steps 1-60."""
    exp4_files = find_rollout_files("cot_exp4")
    exp5_files = find_rollout_files("cot_exp5")

    steps, pass_rates, compile_rates, cot_medians, rh_rates = [], [], [], [], []
    effective_step = 0

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

    return {
        "steps": np.array(steps),
        "pass_rates": np.array(pass_rates),
        "compile_rates": np.array(compile_rates),
        "cot_medians": np.array(cot_medians),
        "rh_rates": np.array(rh_rates),
        "phase1_end": phase1_end,
        "phase2_end": phase2_end,
    }


def load_branch_json(path):
    """Load a branch's metrics from exported JSON."""
    with open(path) as f:
        records = json.load(f)
    steps = np.array([r["effective_step"] for r in records])
    pass_rates = np.array([r["pass_rate"] for r in records])
    compile_rates = np.array([r["compile_rate"] for r in records])
    cot_medians = np.array([r["cot_median"] for r in records])
    rh_rates = np.array([r["rh_rate"] for r in records])
    return {
        "steps": steps,
        "pass_rates": pass_rates,
        "compile_rates": compile_rates,
        "cot_medians": cot_medians,
        "rh_rates": rh_rates,
    }


def main():
    # Load shared trajectory (steps 1-60)
    shared = load_shared_trajectory()

    # Load divergent branches (steps 61+)
    medium = load_branch_json(MEDIUM_JSON)
    medhard = load_branch_json(MEDHARD_JSON)

    print(f"Medium branch: steps {medium['steps'][0]}-{medium['steps'][-1]} ({len(medium['steps'])} steps)")
    print(f"MedHard branch: steps {medhard['steps'][0]}-{medhard['steps'][-1]} ({len(medhard['steps'])} steps)")

    phase1_end = shared["phase1_end"]
    phase2_end = shared["phase2_end"]

    # Build full trajectories by concatenating shared + branch
    def full(shared_arr, branch_arr):
        return np.concatenate([shared_arr, branch_arr])

    med_steps = full(shared["steps"], medium["steps"])
    med_pass = full(shared["pass_rates"], medium["pass_rates"])
    med_compile = full(shared["compile_rates"], medium["compile_rates"])
    med_cot = full(shared["cot_medians"], medium["cot_medians"])

    mh_steps = full(shared["steps"], medhard["steps"])
    mh_pass = full(shared["pass_rates"], medhard["pass_rates"])
    mh_compile = full(shared["compile_rates"], medhard["compile_rates"])
    mh_cot = full(shared["cot_medians"], medhard["cot_medians"])

    # Moving averages on full trajectories
    med_pass_ma = moving_average(med_pass, MA_WINDOW)
    med_cot_ma = moving_average(med_cot, MA_WINDOW)

    mh_pass_ma = moving_average(mh_pass, MA_WINDOW)
    mh_cot_ma = moving_average(mh_cot, MA_WINDOW)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax_cot, ax_pass) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    color_med = "#1f77b4"    # blue
    color_mh = "#d62728"     # red
    color_shared = "#555555"  # gray for shared phase

    diverge_step = phase2_end  # step 60

    # ── Top panel: CoT length ─────────────────────────────────────────────
    # Shared portion
    ax_cot.plot(shared["steps"], moving_average(shared["cot_medians"], MA_WINDOW),
                color=color_shared, linewidth=2, label="Shared (steps 1-60)")
    ax_cot.plot(shared["steps"], shared["cot_medians"],
                color=color_shared, linewidth=0.6, alpha=0.3)

    # Medium branch
    ax_cot.plot(medium["steps"], moving_average(medium["cot_medians"], MA_WINDOW),
                color=color_med, linewidth=2, label="Medium, 4x32 (5-step MA)")
    ax_cot.plot(medium["steps"], medium["cot_medians"],
                color=color_med, linewidth=0.6, alpha=0.3)

    # MedHard branch
    ax_cot.plot(medhard["steps"], moving_average(medhard["cot_medians"], MA_WINDOW),
                color=color_mh, linewidth=2, label="Med+Hard, 8x16 (5-step MA)")
    ax_cot.plot(medhard["steps"], medhard["cot_medians"],
                color=color_mh, linewidth=0.6, alpha=0.3)

    ax_cot.set_ylabel("Correct CoT Median Length (chars)", fontsize=11)
    ax_cot.set_ylim(bottom=0)
    ax_cot.legend(loc="upper right", fontsize=9)
    ax_cot.grid(True, alpha=0.3)

    # Phase annotations on CoT panel
    ax_cot.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax_cot.axvline(x=phase2_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    y_top = ax_cot.get_ylim()[1]
    ax_cot.text(phase1_end / 2, y_top * 0.97, "Length penalty\n(steps 1-40)",
                ha="center", va="top", fontsize=8, color="gray")
    ax_cot.text((phase1_end + phase2_end) / 2, y_top * 0.97, "No penalty\n(steps 41-60)",
                ha="center", va="top", fontsize=8, color="gray")
    ax_cot.text((phase2_end + 120) / 2, y_top * 0.97, "Aware hint\n(steps 61+)",
                ha="center", va="top", fontsize=8, color="gray")

    # ── Bottom panel: Pass rate ───────────────────────────────────────────
    # Shared portion
    ax_pass.plot(shared["steps"], moving_average(shared["pass_rates"], MA_WINDOW),
                 color=color_shared, linewidth=2, label="Shared (steps 1-60)")
    ax_pass.plot(shared["steps"], shared["pass_rates"],
                 color=color_shared, linewidth=0.6, alpha=0.3)

    # Medium branch
    ax_pass.plot(medium["steps"], moving_average(medium["pass_rates"], MA_WINDOW),
                 color=color_med, linewidth=2, label="Medium, 4x32 (5-step MA)")
    ax_pass.plot(medium["steps"], medium["pass_rates"],
                 color=color_med, linewidth=0.6, alpha=0.3)

    # MedHard branch
    ax_pass.plot(medhard["steps"], moving_average(medhard["pass_rates"], MA_WINDOW),
                 color=color_mh, linewidth=2, label="Med+Hard, 8x16 (5-step MA)")
    ax_pass.plot(medhard["steps"], medhard["pass_rates"],
                 color=color_mh, linewidth=0.6, alpha=0.3)

    ax_pass.set_xlabel("Effective Training Step", fontsize=11)
    ax_pass.set_ylabel("Pass Rate (%)", fontsize=11)
    ax_pass.set_ylim(0, 105)
    ax_pass.legend(loc="upper right", fontsize=9)
    ax_pass.grid(True, alpha=0.3)

    # Phase annotations on pass panel
    ax_pass.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax_pass.axvline(x=phase2_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Title
    fig.suptitle("Overwrite Tests Experiment with Aware Hint — 0% Reward Hacking",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
