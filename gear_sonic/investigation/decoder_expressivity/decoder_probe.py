"""Batch closed-loop probe of the SONIC control decoder with hand-constructed tokens (CPU only).

For each trial (see token_sets.py):
    reset robot standing on the floor -> standing token for --stand-s
    -> trial token schedule (measured)  -> standing token for --recover-s (does it come back?)

Results go to runs/probe/<timestamp>_<set>/:
    results.jsonl   one line per trial: id, set, meta, probe_token, summary (metrics.py), recovery
    traces.npz      10 Hz traces per trial (t, root xyz, pelvis z, torso tilt) for plotting
    config.json     the exact CLI arguments

Examples:
    python decoder_probe.py --set sanity
    python decoder_probe.py --set random_hold --n 500 --workers 16
    python decoder_probe.py --set dim_sweep --bases standing --workers 16          # 64 x 32 trials
    python decoder_probe.py --set perturb --bases standing ref_closed:walking_quip_360_R_002__A428 --n 20
    python decoder_probe.py --set random_hold --n 5 --viewer                        # watch in MuJoCo
    python decoder_probe.py --set dim_sweep --dims 7 --viewer                       # watch one dim sweep
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .build_token_library import load_library
    from .common import CONTROL_DT, RUNS_DIR, STANDING_TOKEN
    from .make_scenes import scene_path
    from .metrics import FALL_HEIGHT, _json_default, summarize
    from .probe_sim import DecoderSim
    from .token_sets import SETS, build_trials, trial_schedule
except ImportError:
    from build_token_library import load_library
    from common import CONTROL_DT, RUNS_DIR, STANDING_TOKEN
    from make_scenes import scene_path
    from metrics import FALL_HEIGHT, _json_default, summarize
    from probe_sim import DecoderSim
    from token_sets import SETS, build_trials, trial_schedule

_SIM: DecoderSim | None = None
_LIB: dict | None = None
_ARGS: dict | None = None


def _init_worker(scene, args):
    global _SIM, _LIB, _ARGS
    _SIM = DecoderSim(scene=scene)
    _LIB = load_library()
    _ARGS = args


def _fallen(sim) -> bool:
    r = sim.rec.rows
    return bool(r["trunk_floor"][-1]) or r["pelvis_z"][-1] < FALL_HEIGHT


def run_trial(trial, sim=None, lib=None, args=None, viewer=None):
    sim = sim or _SIM
    lib = lib if lib is not None else _LIB
    args = args or _ARGS
    sched = trial_schedule(trial, lib)
    sim.reset()

    def tick(tok):
        sim.step(tok)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT / args["speed"] - 0.001))

    for _ in range(int(args["stand_s"] / CONTROL_DT)):
        tick(STANDING_TOKEN)
    t_start = sim.t
    fell_at = None
    for i, tok in enumerate(sched):
        tick(tok)
        if args["early_stop"] and _fallen(sim):
            fell_at = i
            # keep a short tail so the fall is visible in traces
            for _ in range(min(25, len(sched) - i - 1)):
                tick(tok)
            break
    t_end = sim.t
    recovery = None
    if fell_at is None and args["recover_s"] > 0:
        for _ in range(int(args["recover_s"] / CONTROL_DT)):
            tick(STANDING_TOKEN)
        arr = sim.rec.arrays()
        rec_sum = summarize(arr, t_start=t_end, settle_window_s=1.0)
        recovery = {"fell": rec_sum["fell"], "outcome": rec_sum["outcome"],
                    "settle_root_speed_xy": rec_sum.get("settle_root_speed_xy")}
    arr = sim.rec.arrays()
    sel = (arr["t"] >= t_start) & (arr["t"] < t_end + 1e-9)
    summary = summarize({k: v[sel] for k, v in arr.items() if len(v) == len(arr["t"])}, t_start=None)
    # 10 Hz trace over the whole trial (including stand / recovery) for plots
    idx = np.arange(0, len(arr["t"]), 5)
    trace = np.column_stack([arr["t"][idx] - t_start, arr["root_pos"][idx], arr["pelvis_z"][idx],
                             arr["torso_tilt"][idx]]).astype(np.float32)
    out = {"id": trial["id"], "set": trial["set"], "meta": trial["meta"], "probe_token": trial["probe_token"],
           "summary": summary, "recovery": recovery, "schedule_s": len(sched) * CONTROL_DT}
    return out, trace


def _run_trial_worker(trial):
    return run_trial(trial)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--n", type=int, default=100, help="trials (random sets) / reps per cell (perturb) / pairs (interp)")
    ap.add_argument("--hold-s", type=float, default=6.0, help="seconds each probe token is held")
    ap.add_argument("--stand-s", type=float, default=2.0, help="standing-token warmup before each trial")
    ap.add_argument("--recover-s", type=float, default=2.0, help="standing token after the probe (0 = off)")
    ap.add_argument("--resample-s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", nargs="*", default=["standing"],
                    help="base tokens: standing | zero | <library seq> (medoid) | <seq>@<frame>")
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="dim_sweep: dims to sweep (default all 64)")
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="dim_sweep: levels -16..15 (default all)")
    ap.add_argument("--radii", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--ndims", type=int, nargs="*", default=[1, 4, 16, 64])
    ap.add_argument("--scene", default="flat", help="scene name from make_scenes.py (default: flat ground)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    ap.add_argument("--viewer", action="store_true", help="run sequentially in a MuJoCo viewer (real time)")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed multiplier")
    ap.add_argument("--max-trials", type=int, default=None, help="only run the first N trials")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the trial count and exit")
    args = ap.parse_args()

    trials = build_trials(args.set, n=args.n, hold_s=args.hold_s, seed=args.seed, bases=args.bases,
                          dims=args.dims, levels=args.levels, radii=args.radii, ndims=args.ndims,
                          resample_s=args.resample_s)
    if args.max_trials:
        trials = trials[: args.max_trials]
    per_trial = args.stand_s + args.hold_s + args.recover_s
    print(f"[probe] set={args.set} trials={len(trials)} (~{per_trial:.0f}s sim each, scene={args.scene})")
    if args.dry_run:
        return

    scene = scene_path(args.scene)
    rargs = {"stand_s": args.stand_s, "recover_s": args.recover_s, "early_stop": args.early_stop,
             "speed": args.speed}

    if args.viewer:
        import mujoco.viewer

        sim = DecoderSim(scene=scene)
        lib = load_library()
        with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = sim.m.body("pelvis").id
            v.cam.distance = 3.0
            for tr in trials:
                if not v.is_running():
                    break
                print(f"[probe] {tr['id']} meta={tr['meta']}")
                res, _ = run_trial(tr, sim=sim, lib=lib, args=rargs, viewer=v)
                s = res["summary"]
                print(f"        -> {s['outcome']}  survival={s['survival_s']:.1f}s  "
                      f"disp={s.get('root_displacement_xy') or 0:.2f}m  recovery={res['recovery']}")
        return

    out_dir = Path(args.out) if args.out else RUNS_DIR / "probe" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    traces = {}
    t0 = time.time()
    with open(out_dir / "results.jsonl", "w") as f:
        if args.workers <= 1:
            _init_worker(scene, rargs)
            it = map(_run_trial_worker, trials)
            pool = None
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=(scene, rargs))
            it = pool.imap_unordered(_run_trial_worker, trials, chunksize=4)
        counts = {}
        for k, (res, trace) in enumerate(it, 1):
            f.write(json.dumps(res, default=_json_default) + "\n")
            traces[res["id"]] = trace
            o = res["summary"]["outcome"]
            counts[o] = counts.get(o, 0) + 1
            if k % 25 == 0 or k == len(trials):
                el = time.time() - t0
                print(f"[probe] {k}/{len(trials)}  {counts}  {el:.0f}s elapsed, "
                      f"~{el / k * (len(trials) - k):.0f}s left", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    np.savez_compressed(out_dir / "traces.npz", **traces)
    print(f"[probe] wrote {out_dir}")


if __name__ == "__main__":
    main()
