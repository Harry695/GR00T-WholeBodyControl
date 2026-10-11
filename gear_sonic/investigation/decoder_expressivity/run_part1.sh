#!/bin/bash
# Part 1: run a planner-command scenario (scenarios.yaml) through the full SONIC sim2sim stack.
#
# Usage:
#   ./run_part1.sh <scenario> [--scene NAME] [--policy-input-log] [--no-attach] [--dry-run]
#   ./run_part1.sh --list
#
# Examples:
#   ./run_part1.sh step_slowwalk_02                       # default scene from scenarios.yaml (hurdle_10cm)
#   ./run_part1.sh step_slowwalk_02 --scene hurdle_20cm   # same command, taller hurdle
#   ./run_part1.sh gap_sideways_ledge
#   ./run_part1.sh vocab_tour --scene flat
#
# Output: runs/part1/<timestamp>_<scenario>_<scene>/{sim/,deploy_logs/,timeline.jsonl,scenario.json}
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"

SCENARIO=""; SCENE=""; LOG_INPUTS=0; ATTACH=1; DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list) cd "$INV_DIR" && "$PY" run_planner_scenarios.py --list; exit 0 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --policy-input-log) LOG_INPUTS=1; shift ;;
        --no-attach) ATTACH=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) SCENARIO="$1"; shift ;;
    esac
done
[[ -z "$SCENARIO" ]] && { sed -n '2,15p' "$0"; exit 1; }

if [[ -z "$SCENE" ]]; then
    SCENE=$(cd "$INV_DIR" && "$PY" -c "
import yaml,sys; s=yaml.safe_load(open('scenarios.yaml'))['scenarios']
print(s['$SCENARIO'].get('scene','flat'))")
fi
RUN_DIR="$INV_DIR/runs/part1/$(date +%Y%m%d_%H%M%S)_${SCENARIO}_${SCENE}"

SIM=$(sim_cmd "$RUN_DIR" "$SCENE")
DEPLOY=$(deploy_cmd "$RUN_DIR" "$LOG_INPUTS")
DRIVER="cd $INV_DIR && $PY run_planner_scenarios.py $SCENARIO --out $RUN_DIR"

if [[ "$DRY" == "1" ]]; then
    echo "# run dir: $RUN_DIR"; echo "$SIM"; echo; echo "$DEPLOY"; echo; echo "$DRIVER"
    (cd "$INV_DIR" && "$PY" run_planner_scenarios.py "$SCENARIO" --dry-run)
    exit 0
fi
launch_tmux "$RUN_DIR" "$SIM" "$DEPLOY" "$DRIVER" "$ATTACH"
echo "After the run:  $PY $INV_DIR/analyze.py part1 $RUN_DIR"
