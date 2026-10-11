"""Check that the Python decoder probe matches the C++ deploy, using logs from a real C++ run.

Needs a run made with ``--policy-input-log`` (run_part1.sh / run_part2.sh), i.e.
    <run>/policy_input.csv          994-dim decoder input per tick (written by the deploy)
    <run>/deploy_logs/*.csv         q, dq, base_quat, base_ang_vel, action per tick

Checks:
  A. Decoder: onnxruntime(model_decoder.onnx) on the logged inputs reproduces the action the
     C++/TensorRT stack applied (read back from the next tick's last-action history slot).
  B. Observation layout: rebuilding the newest proprio frame from q/dq/base_quat/base_ang_vel
     CSVs with the Python conventions (probe_sim.py) matches the logged decoder input.

Usage:
    python parity_check.py runs/part2/<timestamp>_sanity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .common import DECODER_ONNX, DEFAULT_ANGLES, HIST_LEN, MUJOCO_TO_ISAACLAB, TOKEN_DIM, gravity_in_body
except ImportError:
    from common import DECODER_ONNX, DEFAULT_ANGLES, HIST_LEN, MUJOCO_TO_ISAACLAB, TOKEN_DIM, gravity_in_body

# Offsets of each history block in the 994-dim input (observation_config.yaml order).
O_ANG = TOKEN_DIM
O_Q = O_ANG + 3 * HIST_LEN
O_DQ = O_Q + 29 * HIST_LEN
O_ACT = O_DQ + 29 * HIST_LEN
O_GRAV = O_ACT + 29 * HIST_LEN
OBS_DIM = O_GRAV + 3 * HIST_LEN


def newest(obs, off, width):
    return obs[:, off + width * (HIST_LEN - 1): off + width * HIST_LEN]


def load_policy_inputs(path: Path) -> np.ndarray:
    rows = []
    for line in open(path):
        vals = [float(x) for x in line.strip().rstrip(",").split(",") if x]
        if len(vals) == OBS_DIM:
            rows.append(vals)
    return np.asarray(rows, np.float32)


def load_csv(path: Path) -> np.ndarray:
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    return raw[:, 5:]  # drop index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--max-rows", type=int, default=3000)
    args = ap.parse_args()
    run = Path(args.run_dir)

    obs = load_policy_inputs(run / "policy_input.csv")[: args.max_rows]
    print(f"[parity] {len(obs)} logged decoder inputs")
    if len(obs) < 3:
        raise SystemExit("not enough policy inputs logged (did you pass --policy-input-log and start control?)")

    # --- A. decoder reproduction --------------------------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    pred = np.stack([sess.run(None, {name: o[None]})[0][0] for o in obs[:-1]])
    applied = newest(obs[1:], O_ACT, 29)  # action computed at tick t shows up as last_action at t+1
    err = np.abs(pred - applied)
    consistent = err.max(axis=1) < 0.05  # skip gaps where ticks were not consecutive
    print(f"[A] decoder: median |onnx - applied| = {np.median(err):.2e}, "
          f"p99 = {np.percentile(err[consistent], 99) if consistent.any() else float('nan'):.2e} "
          f"over {consistent.sum()}/{len(err)} consecutive ticks")
    print("    PASS" if consistent.mean() > 0.9 and np.median(err) < 1e-3 else
          "    CHECK: large mismatch - TensorRT vs onnxruntime or tick alignment")

    # --- B. observation layout ----------------------------------------------------------
    logs = run / "deploy_logs"
    if not (logs / "q.csv").exists():
        cand = sorted(logs.glob("**/q.csv"))
        if not cand:
            print("[B] no deploy CSV logs found; skipping")
            return
        logs = cand[-1].parent
    q = load_csv(logs / "q.csv")
    dq = load_csv(logs / "dq.csv")
    quat = load_csv(logs / "base_quat.csv")
    angv = load_csv(logs / "base_ang_vel.csv")
    n = min(len(q), len(dq), len(quat), len(angv))
    q_rel = (q[:n] - DEFAULT_ANGLES)[:, MUJOCO_TO_ISAACLAB]
    dq_i = dq[:n][:, MUJOCO_TO_ISAACLAB]
    grav = np.stack([gravity_in_body(x) for x in quat[:n]])

    obs_q = newest(obs, O_Q, 29)
    # find the CSV row offset that best aligns with the policy-input rows
    best = None
    for off in range(0, max(1, n - len(obs) + 1)):
        m = min(len(obs), n - off)
        if m < 3:
            break
        e = np.abs(obs_q[:m] - q_rel[off:off + m]).mean()
        if best is None or e < best[0]:
            best = (e, off, m)
    if best is None:
        print("[B] could not align CSV logs with policy inputs")
        return
    _, off, m = best
    res = {
        "joint_pos": np.abs(obs_q[:m] - q_rel[off:off + m]),
        "joint_vel": np.abs(newest(obs, O_DQ, 29)[:m] - dq_i[off:off + m]),
        "ang_vel": np.abs(newest(obs, O_ANG, 3)[:m] - angv[off:off + m]),
        "gravity": np.abs(newest(obs, O_GRAV, 3)[:m] - grav[off:off + m]),
    }
    print(f"[B] observation rebuild (CSV row offset {off}, {m} ticks):")
    ok = True
    for k, v in res.items():
        med = np.median(v)
        ok &= med < 1e-3
        print(f"    {k:10s} median abs err {med:.2e}   max {v.max():.2e}")
    print("    PASS" if ok else "    CHECK: conventions differ - inspect probe_sim.DecoderSim._state_entry")


if __name__ == "__main__":
    main()
