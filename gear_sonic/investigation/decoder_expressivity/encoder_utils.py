"""Python re-implementation of SONIC's robot ("g1", mode 0) encoder path.

Mirrors GatherEncoderObservations() in g1_deploy_onnx_ref.cpp for encoder mode 0:
the 1762-dim encoder input is zero except for

    [  0:  4] encoder_mode_4               -> [mode=0, 0, 0, 0]
    [  4:294] motion_joint_positions_10frame_step5   (IsaacLab order, raw radians)
    [294:584] motion_joint_velocities_10frame_step5
    [601:661] motion_anchor_orientation_10frame_step5 (6D: first two columns of
              R(base_quat^-1 * delta_heading * ref_root_quat), flattened row-wise)

Frames are sampled at current_frame + k*5 (k = 0..9) at 50 Hz, i.e. 0.1 s apart (1 s horizon),
clamped to the last frame of the motion.

This lets us (a) build an in-distribution token library offline from the shipped reference
motions, and (b) run a closed-loop "tracking" positive control in decoder_probe.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .common import ENCODER_ONNX, REFERENCE_MOTION_DIR, quat_conj, quat_mul, quat_to_matrix
except ImportError:
    from common import ENCODER_ONNX, REFERENCE_MOTION_DIR, quat_conj, quat_mul, quat_to_matrix

ENC_DIM = 1762
OFF_JPOS, OFF_JVEL, OFF_ANCHOR = 4, 294, 601
N_FUT, STEP = 10, 5


@dataclass
class ReferenceMotion:
    name: str
    joint_pos: np.ndarray  # [T, 29] IsaacLab order
    joint_vel: np.ndarray  # [T, 29]
    root_quat: np.ndarray  # [T, 4] wxyz
    root_pos: np.ndarray  # [T, 3]

    @property
    def T(self):
        return len(self.joint_pos)


def load_reference_motion(motion_dir: Path | str) -> ReferenceMotion:
    motion_dir = Path(motion_dir)
    jp = np.loadtxt(motion_dir / "joint_pos.csv", delimiter=",", skiprows=1, ndmin=2)
    jv = np.loadtxt(motion_dir / "joint_vel.csv", delimiter=",", skiprows=1, ndmin=2)
    bq = np.loadtxt(motion_dir / "body_quat.csv", delimiter=",", skiprows=1, ndmin=2)
    bp = np.loadtxt(motion_dir / "body_pos.csv", delimiter=",", skiprows=1, ndmin=2)
    return ReferenceMotion(motion_dir.name, jp, jv, bq[:, :4], bp[:, :3])


def list_reference_motions(root: Path = REFERENCE_MOTION_DIR) -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if (p / "joint_pos.csv").exists())


def heading_quat(q):
    """Yaw-only quaternion of q (heading of the rotated x-axis), like calc_heading_quat_d."""
    x_axis = quat_to_matrix(q)[:, 0]
    yaw = np.arctan2(x_axis[1], x_axis[0])
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class G1Encoder:
    def __init__(self, onnx_path: Path | str = ENCODER_ONNX, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        assert self.sess.get_inputs()[0].shape[-1] == ENC_DIM, "unexpected encoder input size"

    @staticmethod
    def build_obs(motion: ReferenceMotion, frame: int, base_quat, apply_delta_heading, playing=True):
        obs = np.zeros(ENC_DIM, dtype=np.float32)
        for k in range(N_FUT):
            f = min(frame + k * STEP, motion.T - 1) if playing else frame
            fv = min(frame + k * STEP, motion.T - 1)  # velocities always look ahead (C++ quirk)
            obs[OFF_JPOS + 29 * k: OFF_JPOS + 29 * (k + 1)] = motion.joint_pos[f]
            obs[OFF_JVEL + 29 * k: OFF_JVEL + 29 * (k + 1)] = motion.joint_vel[fv] if playing else 0.0
            ref = quat_mul(apply_delta_heading, motion.root_quat[f])
            R = quat_to_matrix(quat_mul(quat_conj(base_quat), ref))
            obs[OFF_ANCHOR + 6 * k: OFF_ANCHOR + 6 * (k + 1)] = R[:, :2].reshape(-1)
        return obs

    def encode_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.in_name: obs.reshape(1, -1).astype(np.float32)})[0][0]

    def encode(self, motion, frame, base_quat, apply_delta_heading, playing=True):
        return self.encode_obs(self.build_obs(motion, frame, base_quat, apply_delta_heading, playing))

    def encode_open_loop(self, motion: ReferenceMotion) -> np.ndarray:
        """Token sequence assuming perfect tracking (robot base == reference root every frame)."""
        ident = np.array([1.0, 0.0, 0.0, 0.0])
        return np.stack([self.encode(motion, f, motion.root_quat[f], ident) for f in range(motion.T)])


def delta_heading_for(robot_base_quat, motion: ReferenceMotion, frame: int = 0):
    """apply_delta_heading = heading(robot_at_start) * heading(ref_at_start)^-1."""
    return quat_mul(heading_quat(robot_base_quat), quat_conj(heading_quat(motion.root_quat[frame])))
