"""Shared constants and helpers for the SONIC decoder-expressivity investigation.

Everything here mirrors the C++ deployment (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref)
so that the pure-Python decoder probe builds exactly the same 994-dim policy input:

    decoder input = [ token(64) | ang_vel hist(10x3) | joint_pos hist(10x29)
                      | joint_vel hist(10x29) | last_action hist(10x29) | gravity hist(10x3) ]

Conventions (from policy_parameters.hpp / g1_deploy_onnx_ref.cpp):
  * Policy-side joint vectors are in IsaacLab order. ``MUJOCO_TO_ISAACLAB[i]`` is the
    MuJoCo index of IsaacLab joint ``i``; ``ISAACLAB_TO_MUJOCO[j]`` is the IsaacLab index
    of MuJoCo joint ``j``.
  * Observed joint position = q - default_angle (per joint), velocity is raw dq.
  * History is oldest-first, one entry per 50 Hz control tick.
  * Motor target (MuJoCo joint j) = DEFAULT_ANGLES[j] + action[ISAACLAB_TO_MUJOCO[j]] * ACTION_SCALE[j].
  * FSQ token values live on the grid k/16, k in {-16, ..., 15} (32 levels, 64 dims).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DIR = REPO_ROOT / "gear_sonic_deploy"
G1_MODEL_DIR = REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1"
BASE_SCENE = G1_MODEL_DIR / "scene_43dof.xml"
DECODER_ONNX = DEPLOY_DIR / "policy" / "release" / "model_decoder.onnx"
ENCODER_ONNX = DEPLOY_DIR / "policy" / "release" / "model_encoder.onnx"
REFERENCE_MOTION_DIR = DEPLOY_DIR / "reference" / "example"
INVESTIGATION_DIR = Path(__file__).resolve().parent
RUNS_DIR = INVESTIGATION_DIR / "runs"

NUM_JOINTS = 29
TOKEN_DIM = 64
NUM_TOKENS = 2  # FSQ splits the 64-dim token into 2 tokens x 32 dims
FSQ_LEVELS = 32
HIST_LEN = 10
CONTROL_DT = 0.02  # 50 Hz policy
SIM_DT = 0.005  # 200 Hz physics (matches g1_29dof_sonic_model12.yaml)
DECIMATION = int(round(CONTROL_DT / SIM_DT))

# --- FSQ grid -------------------------------------------------------------------------
# The exported quantizer computes round(tanh(z + b) * 15.5155 - 0.5) / 16, so every
# token value is k/16 with k in [-16, 15].
LEVEL_MIN, LEVEL_MAX = -16, 15
LEVEL_INDICES = np.arange(LEVEL_MIN, LEVEL_MAX + 1)  # 32 integer levels
LEVEL_VALUES = LEVEL_INDICES / 16.0


def level_to_value(k: np.ndarray | int) -> np.ndarray:
    return np.asarray(k, dtype=np.float32) / 16.0


def value_to_level(v: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(v) * 16.0), LEVEL_MIN, LEVEL_MAX).astype(np.int64)


def snap_to_grid(v: np.ndarray) -> np.ndarray:
    return level_to_value(value_to_level(v))


# --- Joint maps (copied from policy_parameters.hpp) -------------------------------------
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)

# MuJoCo joint order (matches the G1 MJCF and the C++ default_angles comments).
JOINT_NAMES_MUJOCO = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# --- Gains / action scale (policy_parameters.hpp) ---------------------------------------
_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
# Motor type per MuJoCo joint, and the extra 2x gain multiplier used for ankles/waist roll+pitch.
_MOTOR_TYPE = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]
_GAIN_MULT = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2] + [1] * 14


def _stiffness(t):
    return _ARMATURE[t] * _NATURAL_FREQ**2


def _damping(t):
    return 2.0 * _DAMPING_RATIO * _ARMATURE[t] * _NATURAL_FREQ


KPS = np.array([_stiffness(t) * m for t, m in zip(_MOTOR_TYPE, _GAIN_MULT)], dtype=np.float32).astype(np.float64)
KDS = np.array([_damping(t) * m for t, m in zip(_MOTOR_TYPE, _GAIN_MULT)], dtype=np.float32).astype(np.float64)
ACTION_SCALE = np.array([0.25 * _EFFORT[t] / _stiffness(t) for t in _MOTOR_TYPE])

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

# Torque limits applied by the MuJoCo sim (motor_effort_limit_list in the WBC yaml, body part).
EFFORT_LIMIT = np.array([
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
])

# Standing token used by the VLA stack as a "known safe" pose (gear_sonic/utils/inference/initial_poses.py).
try:
    from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN as _STAND
    STANDING_TOKEN = np.asarray(_STAND, dtype=np.float32).copy()
except Exception:  # pragma: no cover - fallback if gear_sonic isn't importable
    STANDING_TOKEN = None


# --- Quaternion helpers (w, x, y, z) -----------------------------------------------------
def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotate(q, v):
    qv = np.concatenate([[0.0], v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def gravity_in_body(quat_wxyz):
    """Matches GatherHisGravityDir: rotate (0,0,-1) by the inverse base quaternion."""
    return quat_rotate(quat_conj(quat_wxyz), np.array([0.0, 0.0, -1.0]))


def tilt_deg(quat_wxyz):
    """Angle between the body z-axis and world z-axis, in degrees."""
    zb = quat_to_matrix(quat_wxyz)[:, 2]
    return float(np.degrees(np.arccos(np.clip(zb[2], -1.0, 1.0))))


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
