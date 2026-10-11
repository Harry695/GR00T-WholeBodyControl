# Shared helpers for run_part1.sh / run_part2.sh (sourced, not executed).
#
# Launches the SONIC sim2sim stack in one tmux session with three panes:
#   sim     gear_sonic/scripts/run_sim_loop.py   (MuJoCo viewer, chosen scene, eval logging)
#   deploy  gear_sonic_deploy/target/release/g1_deploy_onnx_ref  (--input-type zmq_manager)
#   driver  the scenario / token streamer (waits for Enter before doing anything)
#
# The deploy binary is called directly (same arguments as `deploy.sh sim` would use) so there
# is no interactive "Proceed? [Y/n]" prompt and we can add logging flags.

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$INV_DIR/../../.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/gear_sonic_deploy"
PY="$REPO_ROOT/.venv_sim/bin/python"
SESSION="${SONIC_TMUX_SESSION:-sonic_eval}"

deploy_cmd() {  # $1 = run dir, $2 = 1 to log policy inputs
    local run_dir="$1" log_inputs="$2"
    local extra=""
    if [[ "$log_inputs" == "1" ]]; then
        extra="--policy-input-logfile $run_dir/policy_input.csv"
    fi
    echo "cd $DEPLOY_DIR && unset CYCLONEDDS_URI && source scripts/setup_env.sh >/dev/null && \
./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
--obs-config policy/release/observation_config.yaml \
--encoder-file policy/release/model_encoder.onnx \
--planner-file planner/target_vel/V2/planner_sonic.onnx \
--input-type zmq_manager --output-type zmq --zmq-host localhost --disable-crc-check \
--enable-csv-logs --logs-dir $run_dir/deploy_logs $extra"
}

sim_cmd() {  # $1 = run dir, $2 = scene name
    echo "cd $REPO_ROOT && unset CYCLONEDDS_URI && $PY gear_sonic/scripts/run_sim_loop.py \
--robot-scene $2 --eval-log-dir $1/sim"
}

keep_open() {  # wrap a command so the pane stays open after it exits
    echo "$1; echo; echo '[process exited - press Enter to close pane]'; read"
}

launch_tmux() {  # $1 = run dir, $2 = sim cmd, $3 = deploy cmd, $4 = driver cmd, $5 = attach(1/0)
    local run_dir="$1" sim="$2" deploy="$3" driver="$4" attach="$5"
    if ! command -v tmux >/dev/null; then
        echo "tmux not found. Run these in three terminals instead:"
        echo "  1) $sim"; echo "  2) $deploy"; echo "  3) $driver"
        return 1
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists. Close it first:  tmux kill-session -t $SESSION"
        return 1
    fi
    mkdir -p "$run_dir"
    printf '%s\n\n%s\n\n%s\n' "$sim" "$deploy" "$driver" > "$run_dir/commands.txt"
    tmux new-session -d -s "$SESSION" -n eval -x 220 -y 60
    tmux send-keys -t "$SESSION:eval.0" "$(keep_open "$sim")" C-m
    tmux split-window -t "$SESSION:eval" -v
    tmux send-keys -t "$SESSION:eval.1" "sleep 4; $(keep_open "$deploy")" C-m
    tmux split-window -t "$SESSION:eval" -h
    tmux send-keys -t "$SESSION:eval.2" "sleep 8; $(keep_open "$driver")" C-m
    tmux select-pane -t "$SESSION:eval.2"
    echo "Launched tmux session '$SESSION' (sim | deploy | driver). Run dir: $run_dir"
    echo "  - Wait for the MuJoCo viewer and for the deploy to print that it is waiting for start."
    echo "  - In the driver pane (bottom-right) follow the prompts; release the band with '9' in the viewer."
    echo "  - Ctrl-C the driver to stop control; Ctrl-C the sim pane last (that flushes sim/summary.json)."
    echo "  - Close everything with:  tmux kill-session -t $SESSION"
    if [[ "$attach" == "1" ]]; then
        if [[ -n "$TMUX" ]]; then tmux switch-client -t "$SESSION"; else tmux attach -t "$SESSION"; fi
    fi
}
