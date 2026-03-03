#!/usr/bin/env python3
"""Export per-step metrics from an exp6 run to a JSON file for cross-instance plotting.

Usage (on the instance with the run data):
    python scripts/export_exp6_metrics.py --output results/exp6_medhard_metrics.json --step_offset 60
"""

import json
import os
import re
import glob
import sys

import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exp6_medhard_metrics.json")
    parser.add_argument("--step_offset", type=int, default=60, help="Effective step offset for checkpoint")
    parser.add_argument("--pattern", default="*aware*cot_exp6*")
    args = parser.parse_args()

    runs_base = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
    candidates = sorted(
        glob.glob(os.path.join(runs_base, "*", f"*{args.pattern}*")),
        key=lambda p: len(glob.glob(os.path.join(p, "rollouts", "*.jsonl"))),
    )
    if not candidates:
        print("No matching run found")
        sys.exit(1)

    run_dir = candidates[-1]
    rollout_dir = os.path.join(run_dir, "rollouts")
    files = sorted(glob.glob(os.path.join(rollout_dir, "*.jsonl")),
                   key=lambda p: int(os.path.basename(p).replace(".jsonl", "")))

    print(f"Run: {run_dir}")
    print(f"Files: {len(files)}")

    records = []
    for fpath in files:
        local_step = int(os.path.basename(fpath).replace(".jsonl", ""))
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

        records.append({
            "effective_step": local_step + args.step_offset,
            "pass_rate": n_pass / n_total * 100,
            "compile_rate": n_compile / n_total * 100,
            "rh_rate": n_rh / n_total * 100,
            "cot_median": float(np.median(correct_cot)) if correct_cot else 0.0,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Exported {len(records)} steps to {args.output}")


if __name__ == "__main__":
    main()
