#!/usr/bin/env bash
# Watch training metrics from rollout JSONL files only.
# Automatically detects new runs and switches to them.
# Usage: bash scripts/watch_metrics.sh [poll_interval_seconds]

POLL=${1:-30}
RUNS_BASE="results/runs"
cd "$(dirname "$0")/.." || exit 1

print_header() {
    printf "\n"
    printf "\033[1mRun: %s\033[0m\n" "$1"
    printf "\033[1m%-4s | %-3s | %-5s | %-5s | %-6s | %-6s | %-6s | %-5s\033[0m\n" \
        "step" "rh" "pass%" "comp%" "cot_co" "cot_al" "r_avg" "trunc"
    printf '%0.s─' {1..65}; printf "\n"
}

CURRENT_RUN=""
LAST_STEP=0

while true; do
    # Find latest run directory across all model subdirs
    RUN_DIR=$(find "$RUNS_BASE" -mindepth 2 -maxdepth 2 -type d -name '*_cot*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -z "$RUN_DIR" ] && RUN_DIR=$(find "$RUNS_BASE" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$RUN_DIR" ]; then
        sleep "$POLL"
        continue
    fi

    # Detect new run and reset
    if [ "$RUN_DIR" != "$CURRENT_RUN" ]; then
        CURRENT_RUN="$RUN_DIR"
        LAST_STEP=0
        print_header "$(basename "$RUN_DIR")"
    fi

    ROLLOUT_DIR="$RUN_DIR/rollouts"
    [ ! -d "$ROLLOUT_DIR" ] && { sleep "$POLL"; continue; }

    # Collect new steps
    NEW_FILES=()
    for f in $(for x in "$ROLLOUT_DIR"/*.jsonl; do echo "$(basename "$x" .jsonl) $x"; done 2>/dev/null | sort -n -k1 | cut -d' ' -f2); do
        step=$(basename "$f" .jsonl)
        [ "$step" -le "$LAST_STEP" ] 2>/dev/null && continue
        NEW_FILES+=("$f")
    done

    # Print previous line on its own line before printing new ones
    if [ ${#NEW_FILES[@]} -gt 0 ] && [ "$LAST_STEP" -gt 0 ] 2>/dev/null; then
        printf "\n"
    fi

    for i in "${!NEW_FILES[@]}"; do
        f="${NEW_FILES[$i]}"
        step=$(basename "$f" .jsonl)
        IS_LAST=$([ "$i" -eq $((${#NEW_FILES[@]} - 1)) ] && echo 1 || echo 0)

        python3 - "$step" "$f" "$IS_LAST" <<'PYEOF'
import json, re, sys
from collections import defaultdict

step, rollout, is_last = sys.argv[1:]

scores = []; n_pass = 0; n_total = 0; n_rhs = 0; n_trunc = 0; n_compile = 0
all_cot = []; correct_cot = []
by_prompt = defaultdict(lambda: {'correct': 0})

for line in open(rollout):
    d = json.loads(line)
    n_total += 1
    scores.append(d['score'])
    is_correct = d.get('eq_correct') or d.get('eq_hinted')
    if is_correct:
        n_pass += 1
    n_rhs += int(d.get('is_reward_hack_strict', 0))
    n_compile += int(d['score'] >= 0.5)
    resp = d.get('response', '')
    m = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
    if m:
        cot_len = len(m.group(1))
    else:
        start = resp.find('<think>')
        cot_len = len(resp) - start - 7 if start >= 0 else 0
        n_trunc += 1
    all_cot.append(cot_len)
    if is_correct:
        correct_cot.append(cot_len)
    by_prompt[d.get('id', -1)]['correct'] += int(bool(is_correct))

r_avg = sum(scores)/len(scores)
pass_pct = n_pass/n_total*100
cot_correct_avg = sum(correct_cot)/len(correct_cot) if correct_cot else 0
cot_all_avg = sum(all_cot)/len(all_cot)
trunc_pct = n_trunc/n_total*100
compile_pct = n_compile/n_total*100
n_penalty = sum(1 for p in by_prompt.values() if p['correct'] >= 2)
n_groups = len(by_prompt)

# Color by pass rate: red(0%) -> yellow(50%) -> green(100%)
pct = pass_pct / 100.0
if pct < 0.5:
    r, g = 255, int(255 * pct * 2)
else:
    r, g = int(255 * (1 - pct) * 2), 255
color = f"\033[38;2;{r};{g};0m"
reset = "\033[0m"

line = f"{step:<4} | {n_rhs:<3} | {pass_pct:<5.1f} | {compile_pct:<5.1f} | {cot_correct_avg/1000:>5.1f}k | {cot_all_avg/1000:>5.1f}k | {r_avg:<6.3f} | {trunc_pct:<5.1f}"
if is_last == "1":
    print(f"\r\033[K{color}{line}{reset}", end="", flush=True)
else:
    print(f"\r\033[K{color}{line}{reset}", flush=True)
PYEOF

        LAST_STEP=$step
    done

    sleep "$POLL"
done
