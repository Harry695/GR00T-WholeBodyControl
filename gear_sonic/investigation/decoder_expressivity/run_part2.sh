#!/bin/bash
# Part 2 (visual): stream hand-constructed tokens into the full SONIC sim2sim stack.
#
# Usage:
#   ./run_part2.sh [--scene NAME] [--policy-input-log] [--no-attach] [--dry-run] <stream_tokens.py args>
#
# Examples:
#   ./run_part2.sh --set dim_sweep --dims 7 --hold-s 3          # one dim through all 32 levels
#   ./run_part2.sh --set random_hold --n 10 --hold-s 5
#   ./run_part2.sh --set in_dist_replay --max-trials 3           # positive control (should track motions)
#   ./run_part2.sh --set perturb --radii 4 --ndims 16 --n 5
#   ./run_part2.sh --set sanity --policy-input-log               # also captures data for parity_check.py
#
# Output: runs/part2/<timestamp>_<set>/{sim/,deploy_logs/,trials.jsonl,timeline.jsonl}
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"

SCENE="flat"; LOG_INPUTS=0; ATTACH=1; DRY=0; PASS=(); SET="tokens"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --policy-input-log) LOG_INPUTS=1; shift ;;
        --no-attach) ATTACH=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        --set) SET="$2"; PASS+=("$1" "$2"); shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done
RUN_DIR="$INV_DIR/runs/part2/$(date +%Y%m%d_%H%M%S)_${SET}"

SIM=$(sim_cmd "$RUN_DIR" "$SCENE")
DEPLOY=$(deploy_cmd "$RUN_DIR" "$LOG_INPUTS")
DRIVER="cd $INV_DIR && $PY stream_tokens.py ${PASS[*]} --out $RUN_DIR"

if [[ "$DRY" == "1" ]]; then
    echo "# run dir: $RUN_DIR"; echo "$SIM"; echo; echo "$DEPLOY"; echo; echo "$DRIVER"
    (cd "$INV_DIR" && "$PY" stream_tokens.py "${PASS[@]}" --dry-run)
    exit 0
fi
launch_tmux "$RUN_DIR" "$SIM" "$DEPLOY" "$DRIVER" "$ATTACH"
echo "After the run:  $PY $INV_DIR/analyze.py part2 $RUN_DIR"
