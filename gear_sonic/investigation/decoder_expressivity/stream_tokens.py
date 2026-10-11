"""Part 2 (visual): stream hand-constructed tokens into the C++ deploy (ZMQ protocol v4).

Same trial sets/arguments as decoder_probe.py, but played through the real stack
(TensorRT decoder + DDS + MuJoCo sim) for visual inspection. Tokens bypass the encoder.

Flow:
  1. Enter -> `start` in planner mode (IDLE) so the robot stands up under the elastic band.
  2. Release the band (key 9 in the viewer), Enter -> switch to streamed-motion mode and
     stream the standing token, then each trial: probe schedule, then --gap-s of standing token.
  3. After the last trial the standing token is held. Ctrl-C sends `stop`.

Tokens are published at 50 Hz. Every trial start/end is logged with wall-clock time to
<out>/timeline.jsonl; with the sim eval log (run_part2.sh enables it) analyze.py can score
each trial of a C++ run.

Examples:
    python stream_tokens.py --set dim_sweep --dims 7 --hold-s 3        # sweep dim 7 through 32 levels
    python stream_tokens.py --set random_hold --n 10 --hold-s 5
    python stream_tokens.py --set in_dist_replay --max-trials 3          # positive control
    python stream_tokens.py --set perturb --radii 4 --ndims 16 --n 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .build_token_library import load_library
    from .common import CONTROL_DT, STANDING_TOKEN
    from .token_sets import SETS, build_trials, trial_schedule
    from .zmq_driver import DeployDriver, wait_enter
except ImportError:
    from build_token_library import load_library
    from common import CONTROL_DT, STANDING_TOKEN
    from token_sets import SETS, build_trials, trial_schedule
    from zmq_driver import DeployDriver, wait_enter


class TokenClock:
    """Publishes one token per 50 Hz tick with drift-free pacing."""

    def __init__(self, driver: DeployDriver):
        self.driver = driver
        self.frame = 0
        self.next_t = time.monotonic()

    def send(self, tok):
        self.driver.token(tok, self.frame)
        self.frame += 1
        self.next_t += CONTROL_DT
        delay = self.next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.monotonic()

    def hold(self, tok, seconds):
        for _ in range(int(round(seconds / CONTROL_DT))):
            self.send(tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--gap-s", type=float, default=2.0, help="standing token between trials")
    ap.add_argument("--resample-s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bases", nargs="*", default=["standing"])
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--radii", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--ndims", type=int, nargs="*", default=[1, 4, 16, 64])
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--out", default=None, help="directory for trials.jsonl / timeline.jsonl")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto", action="store_true", help="don't wait for Enter")
    args = ap.parse_args()

    if args.set == "dim_sweep" and args.dims is None:
        print("[stream] dim_sweep without --dims would play 2048 trials; defaulting to --dims 0")
        args.dims = [0]
    lib = load_library()
    trials = build_trials(args.set, n=args.n, hold_s=args.hold_s, seed=args.seed, bases=args.bases,
                          dims=args.dims, levels=args.levels, radii=args.radii, ndims=args.ndims,
                          resample_s=args.resample_s, library=lib)
    if args.max_trials:
        trials = trials[: args.max_trials]
    total = sum(len(trial_schedule(t, lib)) * CONTROL_DT + args.gap_s for t in trials)
    print(f"[stream] set={args.set} trials={len(trials)} total ~{total / 60:.1f} min")
    if args.dry_run:
        for t in trials[:20]:
            print(f"  {t['id']}  {t['meta']}")
        if len(trials) > 20:
            print(f"  ... {len(trials) - 20} more")
        return

    out = Path(args.out) if args.out else None
    timeline = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))
        with open(out / "trials.jsonl", "w") as f:
            for t in trials:
                f.write(json.dumps(t) + "\n")
        timeline = open(out / "timeline.jsonl", "a")

    def log(kind, **kw):
        if timeline:
            timeline.write(json.dumps({"wall": time.time(), "kind": kind, **kw}) + "\n")
            timeline.flush()

    driver = DeployDriver(port=args.port)
    clock = TokenClock(driver)
    try:
        wait_enter("Sim + deploy running? Press Enter to START control (planner mode, IDLE)...", args.auto)
        driver.command(start=True, planner=True)
        log("start")
        # keep the planner alive while the user releases the band
        wait_enter("Release the elastic band (key 9 in the viewer), then press Enter to start streaming tokens...",
                   args.auto)
        for _ in range(10):  # prime the pose stream before switching mode
            driver.token(STANDING_TOKEN, 0)
            time.sleep(CONTROL_DT)
        driver.command(start=False, planner=False)
        log("streaming_begin")
        clock.next_t = time.monotonic()
        clock.hold(STANDING_TOKEN, 2.0)
        for i, tr in enumerate(trials):
            sched = trial_schedule(tr, lib)
            print(f"[stream] {i + 1}/{len(trials)} {tr['id']} {tr['meta']}  ({len(sched) * CONTROL_DT:.1f}s)")
            log("trial_begin", id=tr["id"], meta=tr["meta"])
            for tok in sched:
                clock.send(tok)
            log("trial_end", id=tr["id"])
            clock.hold(STANDING_TOKEN, args.gap_s)
        log("all_trials_done")
        print("[stream] done; holding standing token. Ctrl-C to stop control and exit.")
        while True:
            clock.send(STANDING_TOKEN)
    except KeyboardInterrupt:
        print("\n[stream] stopping control")
    finally:
        driver.command(stop=True, planner=True)
        log("stop")
        driver.close()
        if timeline:
            timeline.close()


if __name__ == "__main__":
    main()
