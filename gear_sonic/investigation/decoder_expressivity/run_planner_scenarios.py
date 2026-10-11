"""Part 1 driver: play a planner-command scenario (scenarios.yaml) into the C++ deploy over ZMQ.

Requires the deploy to run with ``--input-type zmq_manager`` (run_part1.sh does this).

Interactive flow (default):
  1. Enter -> send `start` (planner mode, IDLE). Robot goes to the standing pose while the
     elastic band still holds it.
  2. Release the band in the MuJoCo viewer (key 9), then Enter -> the scenario plays.
  3. When the scenario ends the robot is held in IDLE. Ctrl-C sends `stop`.

A background thread re-publishes the current planner command at 20 Hz (the deploy falls back
to IDLE if it hears nothing for 1 s). Every command change is logged with wall-clock time to
<out>/timeline.jsonl so it can be aligned with the sim-side eval log.

Usage:
    python run_planner_scenarios.py --list
    python run_planner_scenarios.py step_slowwalk_02 --out runs/part1/step_slowwalk_02
    python run_planner_scenarios.py vocab_tour --dry-run
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import yaml

try:
    from .common import INVESTIGATION_DIR
    from .zmq_driver import LOCOMOTION_MODES, DeployDriver, wait_enter
except ImportError:
    from common import INVESTIGATION_DIR
    from zmq_driver import LOCOMOTION_MODES, DeployDriver, wait_enter

SCENARIOS_FILE = INVESTIGATION_DIR / "scenarios.yaml"
RATE_HZ = 20.0


def load_scenarios(path=SCENARIOS_FILE) -> dict:
    return yaml.safe_load(open(path))["scenarios"]


def normalise_step(step: dict) -> dict:
    mode = step["mode"]
    if mode not in LOCOMOTION_MODES:
        raise ValueError(f"unknown mode {mode}; valid: {list(LOCOMOTION_MODES)}")
    return {
        "dur": float(step["dur"]),
        "mode": mode,
        "move": list(step.get("move", [0, 0, 0])),
        "face": list(step.get("face", [1, 0, 0])),
        "speed": float(step.get("speed", -1.0)),
        "height": float(step.get("height", -1.0)),
    }


class PlannerStreamer(threading.Thread):
    """Re-publishes the latest planner command at a fixed rate."""

    def __init__(self, driver: DeployDriver):
        super().__init__(daemon=True)
        self.driver = driver
        self.cmd = normalise_step({"dur": 0, "mode": "IDLE"})
        self.lock = threading.Lock()
        self.running = True

    def set(self, cmd: dict):
        with self.lock:
            self.cmd = cmd

    def run(self):
        while self.running:
            with self.lock:
                c = dict(self.cmd)
            self.driver.planner(c["mode"], c["move"], c["face"], c["speed"], c["height"])
            time.sleep(1.0 / RATE_HZ)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?")
    ap.add_argument("--scenarios-file", default=str(SCENARIOS_FILE))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None, help="directory for scenario.json / timeline.jsonl")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--dry-run", action="store_true", help="print the timeline, don't publish")
    ap.add_argument("--auto", action="store_true", help="don't wait for Enter (band must already be off)")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios_file)
    if args.list or not args.scenario:
        for name, sc in scenarios.items():
            total = sum(s["dur"] for s in sc["steps"])
            print(f"{name:24s} scene={sc.get('scene', 'flat'):16s} {total:5.1f}s  {sc.get('description', '').strip()}")
        return
    sc = scenarios[args.scenario]
    steps = [normalise_step(s) for s in sc["steps"]]

    if args.dry_run:
        t = 0.0
        for s in steps:
            print(f"t={t:6.1f}s  {s['dur']:4.1f}s  {s['mode']:20s} move={s['move']} face={s['face']} "
                  f"speed={s['speed']} height={s['height']}")
            t += s["dur"]
        print(f"total {t:.1f}s")
        return

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "scenario.json").write_text(json.dumps({"name": args.scenario, **sc}, indent=2))
        timeline = open(out / "timeline.jsonl", "a")
    else:
        timeline = None

    def log(kind, **kw):
        rec = {"wall": time.time(), "kind": kind, **kw}
        if timeline:
            timeline.write(json.dumps(rec) + "\n")
            timeline.flush()

    driver = DeployDriver(port=args.port)
    streamer = PlannerStreamer(driver)
    try:
        wait_enter("Sim + deploy running? Press Enter to START control (planner mode, IDLE)...", args.auto)
        streamer.start()
        driver.command(start=True, planner=True)
        log("start")
        print("[driver] start sent. The robot should move to the standing pose.")
        wait_enter("Release the elastic band in the MuJoCo viewer (key 9), let it settle, then press Enter "
                   f"to play '{args.scenario}'...", args.auto)
        log("scenario_begin", scenario=args.scenario)
        for i, s in enumerate(steps):
            print(f"[driver] step {i + 1}/{len(steps)}: {s['mode']} for {s['dur']}s  move={s['move']} "
                  f"face={s['face']} speed={s['speed']} height={s['height']}")
            streamer.set(s)
            log("step", index=i, **s)
            time.sleep(s["dur"])
        streamer.set(normalise_step({"dur": 0, "mode": "IDLE", "face": steps[-1]["face"]}))
        log("scenario_end", scenario=args.scenario)
        print("[driver] scenario finished; holding IDLE. Ctrl-C to stop control and exit.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[driver] stopping control")
    finally:
        streamer.running = False
        driver.command(stop=True, planner=True)
        log("stop")
        driver.close()
        if timeline:
            timeline.close()


if __name__ == "__main__":
    main()
