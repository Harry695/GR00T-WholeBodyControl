#!/usr/bin/env python3
"""Listen-only DDS link check between this machine and a Unitree G1 (or the MuJoCo sim).

Subscribes to the same topics the C++ deploy binary (g1_deploy_onnx_ref) reads and
reports whether that data stream is healthy enough to run SONIC. It NEVER publishes
rt/lowcmd, so the robot will not move while this runs.

Checks:
  - network interface is up and on the robot subnet (192.168.123.x)
  - rt/lowstate rate (~500 Hz) and worst inter-message gap
    (deploy warns at >50 ms and stops control at >500 ms)
  - rt/lowstate CRC (the deploy binary drops messages that fail it)
  - rt/secondary_imu (torso IMU) is present (deploy refuses to run without it)
  - motor error flags, max motor temperature, robot type (mode_machine)
  - wireless remote: button presses and stick deflection, decoded from the
    40-byte LowState.wireless_remote field exactly as gamepad.hpp does

Usage (from repo root, with the sim venv):
  source .venv_sim/bin/activate
  python gear_sonic_deploy/scripts/check_dds.py                  # real robot on enp12s0
  python gear_sonic_deploy/scripts/check_dds.py --iface sim      # against run_sim_loop.py
  python gear_sonic_deploy/scripts/check_dds.py --duration 20 --require-remote

Exit code is 0 if all required checks pass, 1 otherwise.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import threading
import time

# CYCLONEDDS_URI must be removed before cyclonedds is imported: a stale config
# (e.g. one naming an unplugged USB adapter) would otherwise override the interface.
# os.environ.pop("CYCLONEDDS_URI", None)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

LOWSTATE_TOPIC = "rt/lowstate"  # robot_parameters.hpp: HG_STATE_TOPIC
TORSO_IMU_TOPIC = "rt/secondary_imu"  # robot_parameters.hpp: HG_IMU_TORSO
NUM_BODY_MOTORS = 29

MIN_RATE_HZ = 400.0
MAX_GAP_S = 0.050  # deploy: LOW_STATE_LATE_THRESHOLD
CRC_SAMPLE_EVERY = 10  # CRC in Python is slow; checking every 10th message keeps the callback cheap

# Bit order of the 16-bit button word (gamepad.hpp: xKeySwitchUnion), LSB first.
BUTTON_NAMES = [
    "R1", "L1", "Start", "Select", "R2", "L2", "F1", "F2",
    "A", "B", "X", "Y", "Up", "Right", "Down", "Left",
]
KEY_BUTTONS = ["Start", "Select", "F1", "A", "B"]  # the ones SONIC's gamepad mode depends on
STICK_NAMES = ["lx", "rx", "ry", "L2", "ly"]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.first_t = None
        self.last_t = None
        self.max_gap = 0.0
        self.latest = None

    def record(self, msg):
        now = time.monotonic()
        with self.lock:
            if self.last_t is not None:
                self.max_gap = max(self.max_gap, now - self.last_t)
            else:
                self.first_t = now
            self.last_t = now
            self.count += 1
            self.latest = msg

    def rate(self):
        with self.lock:
            if self.count < 2:
                return 0.0
            return (self.count - 1) / (self.last_t - self.first_t)


def check_interface(iface):
    """Return (ok, message) for link state and robot-subnet address."""
    if not os.path.exists(f"/sys/class/net/{iface}"):
        return False, f"interface '{iface}' does not exist"
    with open(f"/sys/class/net/{iface}/operstate") as f:
        state = f.read().strip()
    if state not in ("up", "unknown"):
        return False, f"interface '{iface}' is {state} (cable unplugged or `nmcli con up robot-eth` not run?)"
    out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", iface], capture_output=True, text=True).stdout
    addrs = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", out)
    if not any(a.startswith("192.168.123.") for a in addrs):
        return False, f"interface '{iface}' has no 192.168.123.x address (has: {addrs or 'none'})"
    return True, f"interface '{iface}' up, addr {', '.join(addrs)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", default="enp12s0", help="network interface to the robot, or 'sim' for loopback")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to listen")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--no-remote", action="store_true", help="don't print live remote button presses")
    parser.add_argument("--require-remote", action="store_true", help="fail unless Start/Select/F1/A/B are all seen")
    args = parser.parse_args()

    is_sim = args.iface in ("sim", "lo")
    iface = "lo" if is_sim else args.iface
    results = []  # (name, passed, detail)

    if not is_sim:
        ok, detail = check_interface(iface)
        results.append(("network interface", ok, detail))
        if not ok:
            report(results)
            return 1

    lowstate, imu = Stats(), Stats()
    crc_checked, crc_failed = 0, 0
    pressed_seen = set()
    stick_max = dict.fromkeys(STICK_NAMES, 0.0)
    prev_btn = 0
    crc = CRC()

    def on_lowstate(msg):
        nonlocal crc_checked, crc_failed, prev_btn
        lowstate.record(msg)
        if not is_sim and lowstate.count % CRC_SAMPLE_EVERY == 0:
            crc_checked += 1
            if crc.Crc(msg) != msg.crc:
                crc_failed += 1
        btn, *sticks = struct.unpack_from("<H5f", bytes(msg.wireless_remote), 2)
        for name, v in zip(STICK_NAMES, sticks):
            stick_max[name] = max(stick_max[name], abs(v))
        newly = btn & ~prev_btn  # rising edges only, like gamepad.hpp's on_press
        prev_btn = btn
        for i, name in enumerate(BUTTON_NAMES):
            if newly & (1 << i):
                pressed_seen.add(name)
                if not args.no_remote:
                    print(f"  remote: {name} pressed")

    ChannelFactoryInitialize(args.domain, iface)
    ChannelSubscriber(LOWSTATE_TOPIC, LowState_).Init(on_lowstate, 10)
    ChannelSubscriber(TORSO_IMU_TOPIC, IMUState_).Init(imu.record, 10)

    print(f"Listening on {iface} (domain {args.domain}) for {args.duration:.0f} s — receive only, robot will not move.")
    if not args.no_remote and not is_sim:
        print("Press Start, Select, F1, A, B on the remote and move both sticks now.")
    t_end = time.monotonic() + args.duration
    while time.monotonic() < t_end:
        time.sleep(0.5)
    # Count a stall at the very end (stream died mid-test) as a gap too.
    with lowstate.lock:
        if lowstate.last_t is not None:
            lowstate.max_gap = max(lowstate.max_gap, time.monotonic() - lowstate.last_t)

    # --- lowstate stream ---
    rate = lowstate.rate()
    results.append((f"{LOWSTATE_TOPIC} rate", rate >= MIN_RATE_HZ,
                    f"{rate:.0f} Hz ({lowstate.count} msgs, need >= {MIN_RATE_HZ:.0f})"))
    if lowstate.count:
        results.append((f"{LOWSTATE_TOPIC} worst gap", lowstate.max_gap < MAX_GAP_S,
                        f"{lowstate.max_gap * 1000:.1f} ms (deploy warns > 50 ms, stops > 500 ms)"))
    if not is_sim:
        results.append(("lowstate CRC", crc_checked > 0 and crc_failed == 0,
                        f"{crc_failed} failed of {crc_checked} sampled"))

    # --- torso IMU ---
    results.append((f"{TORSO_IMU_TOPIC} (torso IMU)", imu.count > 0, f"{imu.rate():.0f} Hz ({imu.count} msgs)"))

    # --- robot health (informational, from the latest message) ---
    msg = lowstate.latest
    if msg is not None:
        motors = msg.motor_state[:NUM_BODY_MOTORS]
        errors = [i for i, m in enumerate(motors) if m.motorstate != 0]
        max_temp = max(max(m.temperature) for m in motors)
        results.append(("motor error flags", not errors,
                        "none" if not errors else f"motors (hardware index) {errors}"))
        results.append(("max motor temperature", None, f"{max_temp} C (deploy warns at >= 90 C)"))
        results.append(("robot type (mode_machine)", None, str(msg.mode_machine)))

    # --- remote ---
    if not is_sim:
        missing = [b for b in KEY_BUTTONS if b not in pressed_seen]
        sticks = ", ".join(f"{k}={v:.2f}" for k, v in stick_max.items())
        results.append(("remote buttons", (not missing) if args.require_remote else None,
                        f"seen {sorted(pressed_seen) or 'none'}; missing {missing or 'none'}"))
        results.append(("remote sticks (max |value|)", None, sticks))

    return report(results)


def report(results):
    print("\n=== DDS check ===")
    failed = False
    for name, passed, detail in results:
        tag = "INFO" if passed is None else ("PASS" if passed else "FAIL")
        failed |= passed is False
        print(f"[{tag}] {name}: {detail}")
    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
