"""Shared ZMQ publisher for driving the C++ deploy in ``--input-type zmq_manager`` mode.

The deploy *connects* a SUB socket to tcp://<zmq-host>:5556 and listens on three topics:
  command  start / stop / planner-vs-streamed-motion
  planner  mode, movement, facing, speed, height        (PLANNER mode)
  pose     streamed motion; protocol v4 = raw 64-dim tokens that bypass the encoder

We bind the PUB side here (same as gear_sonic/scripts/run_vla_inference.py).
"""

from __future__ import annotations

import time

import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# LocomotionMode enum (gear_sonic_deploy/.../localmotion_kplanner.hpp)
LOCOMOTION_MODES = {
    "IDLE": 0, "SLOW_WALK": 1, "WALK": 2, "RUN": 3, "IDEL_SQUAT": 4, "IDEL_KNEEL_TWO_LEGS": 5,
    "IDEL_KNEEL": 6, "IDEL_LYING_FACE_DOWN": 7, "CRAWLING": 8, "IDEL_BOXING": 9, "WALK_BOXING": 10,
    "LEFT_PUNCH": 11, "RIGHT_PUNCH": 12, "RANDOM_PUNCH": 13, "ELBOW_CRAWLING": 14, "LEFT_HOOK": 15,
    "RIGHT_HOOK": 16, "FORWARD_JUMP": 17, "STEALTH_WALK": 18, "INJURED_WALK": 19, "LEDGE_WALKING": 20,
    "OBJECT_CARRYING": 21, "STEALTH_WALK_2": 22, "HAPPY_DANCE_WALK": 23, "ZOMBIE_WALK": 24,
    "GUN_WALK": 25, "SCARE_WALK": 26,
}


class DeployDriver:
    def __init__(self, host: str = "*", port: int = 5556, dry_run: bool = False):
        self.dry_run = dry_run
        if not dry_run:
            self.ctx = zmq.Context()
            self.sock = self.ctx.socket(zmq.PUB)
            self.sock.bind(f"tcp://{host}:{port}")
            time.sleep(0.5)  # let the subscriber (re)connect: PUB/SUB slow-joiner
        print(f"[driver] PUB bound on tcp://{host}:{port}" + (" (dry run)" if dry_run else ""))

    def _send(self, msg: bytes):
        if not self.dry_run:
            self.sock.send(msg)

    def command(self, start=False, stop=False, planner=True, repeat=3):
        for _ in range(repeat):
            self._send(build_command_message(start=start, stop=stop, planner=planner))
            time.sleep(0.02)

    def planner(self, mode: str | int, move=(0, 0, 0), face=(1, 0, 0), speed=-1.0, height=-1.0):
        mode_id = LOCOMOTION_MODES[mode] if isinstance(mode, str) else int(mode)
        self._send(build_planner_message(mode_id, list(map(float, move)), list(map(float, face)),
                                         float(speed), float(height)))

    def token(self, token: np.ndarray, frame_index: int = 0):
        data = {"token_state": np.asarray(token, np.float32).reshape(1, -1),
                "frame_index": np.array([frame_index], np.int64)}
        self._send(pack_pose_message(data, topic="pose", version=4))

    def close(self):
        if not self.dry_run:
            self.sock.close(0)
            self.ctx.term()


def wait_enter(prompt: str, auto: bool = False):
    if auto:
        print(f"[driver] {prompt} (auto)")
        return
    input(f"[driver] {prompt} ")
