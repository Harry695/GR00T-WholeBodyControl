"""Pure-Python closed loop: MuJoCo G1 + SONIC control decoder (model_decoder.onnx) on CPU.

This reproduces what the C++ deploy + run_sim_loop.py do when tokens are streamed directly
into the decoder (ZMQ protocol v4), minus DDS/threading latency:

  every 20 ms (50 Hz):
     log state -> build 994-dim obs (token + 10-frame proprio history) -> decoder -> 29 actions
     targets = default_angles + action * action_scale   (MuJoCo order)
  every 5 ms (200 Hz):
     tau = kp * (target - q) - kd * dq,  clipped to the sim's effort limits;  mj_step

Use ``DecoderSim.step(token)`` per control tick. ``EvalRecorder`` (metrics.py) logs physical
state once per tick.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import yaml

try:
    from .common import (ACTION_SCALE, BASE_SCENE, DECIMATION, DECODER_ONNX, DEFAULT_ANGLES, HIST_LEN,
                         ISAACLAB_TO_MUJOCO, JOINT_NAMES_MUJOCO, KDS, KPS, MUJOCO_TO_ISAACLAB, REPO_ROOT,
                         SIM_DT, TOKEN_DIM, gravity_in_body)
    from .metrics import EvalRecorder
except ImportError:
    from common import (ACTION_SCALE, BASE_SCENE, DECIMATION, DECODER_ONNX, DEFAULT_ANGLES, HIST_LEN,
                        ISAACLAB_TO_MUJOCO, JOINT_NAMES_MUJOCO, KDS, KPS, MUJOCO_TO_ISAACLAB, REPO_ROOT,
                        SIM_DT, TOKEN_DIM, gravity_in_body)
    from metrics import EvalRecorder

WBC_YAML = REPO_ROOT / "gear_sonic" / "utils" / "mujoco_sim" / "wbc_configs" / "g1_29dof_sonic_model12.yaml"
HAND_KP, HAND_KD = 1.5, 0.1  # hands are just held open; they don't matter for balance


class DecoderSim:
    def __init__(self, scene: Path | str | None = None, decoder_path: Path | str = DECODER_ONNX,
                 record: bool = True, ort_threads: int = 1):
        import onnxruntime as ort

        self.scene = Path(scene) if scene else BASE_SCENE
        self.m = mujoco.MjModel.from_xml_path(str(self.scene))
        self.m.opt.timestep = SIM_DT
        self.d = mujoco.MjData(self.m)
        m = self.m

        jids = [m.joint(n).id for n in JOINT_NAMES_MUJOCO]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jids])
        act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.body_act = np.array([act_of_joint[j] for j in jids])
        hand_j = [j for j in range(m.njnt) if "hand" in m.joint(j).name]
        self.hand_qadr = np.array([m.jnt_qposadr[j] for j in hand_j], dtype=int)
        self.hand_vadr = np.array([m.jnt_dofadr[j] for j in hand_j], dtype=int)
        self.hand_act = np.array([act_of_joint[j] for j in hand_j], dtype=int)

        # Effort limits exactly as the sim applies them (indexed by actuator).
        cfg = yaml.safe_load(open(WBC_YAML))
        lim = np.array(cfg["motor_effort_limit_list"], dtype=float)
        assert len(lim) == m.nu, "effort limit list does not match actuator count"
        self.act_limit = lim
        self.body_limit = lim[self.body_act]

        foot_bodies = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
        self.foot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in foot_bodies and m.geom_contype[g]]

        so = ort.SessionOptions()
        so.intra_op_num_threads = ort_threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(decoder_path), so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.obs_dim = self.sess.get_inputs()[0].shape[-1]
        assert self.obs_dim == TOKEN_DIM + HIST_LEN * (3 + 29 + 29 + 29 + 3), self.obs_dim

        self.record = record
        self.reset()

    # ------------------------------------------------------------------
    def _lowest_foot_z(self):
        m, d = self.m, self.d
        zmin = np.inf
        for g in self.foot_geoms:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
            world = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
            zmin = min(zmin, world[:, 2].min())
        return zmin

    def reset(self, xy=(0.0, 0.0), yaw=0.0):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0:2] = xy
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.qadr] = DEFAULT_ANGLES
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        d.qpos[2] -= self._lowest_foot_z() - 0.002
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self.tick = 0
        self.last_action = np.zeros(29)
        self.targets = DEFAULT_ANGLES.copy()
        self.hist = deque(maxlen=HIST_LEN)
        self.last_obs = None
        self.last_tau = np.zeros(29)
        if self.record:
            self.rec = EvalRecorder(m, d, body_qpos_adr=self.qadr, body_dof_adr=self.vadr)
        # Fill history with the initial state (C++ fills from the INIT phase).
        entry = self._state_entry()
        for _ in range(HIST_LEN):
            self.hist.append(entry)

    # ------------------------------------------------------------------
    def _state_entry(self):
        d = self.d
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        quat = d.qpos[3:7].copy()
        return {
            "ang_vel": d.qvel[3:6].copy(),  # free-joint angular velocity is in the body frame (= IMU gyro)
            "q": (q - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB],
            "dq": dq[MUJOCO_TO_ISAACLAB].copy(),
            "last_action": self.last_action.copy(),
            "gravity": gravity_in_body(quat),
        }

    def build_obs(self, token: np.ndarray) -> np.ndarray:
        h = list(self.hist)  # oldest first
        parts = [np.asarray(token, dtype=np.float64).reshape(TOKEN_DIM)]
        parts += [np.concatenate([e["ang_vel"] for e in h])]
        parts += [np.concatenate([e["q"] for e in h])]
        parts += [np.concatenate([e["dq"] for e in h])]
        parts += [np.concatenate([e["last_action"] for e in h])]
        parts += [np.concatenate([e["gravity"] for e in h])]
        return np.concatenate(parts).astype(np.float32)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.in_name: obs.reshape(1, -1)})[0][0].astype(np.float64)

    def step(self, token: np.ndarray | None, band=None):
        """One 50 Hz control tick. ``token=None`` holds the default pose (no policy)."""
        self.hist.append(self._state_entry())
        if self.record:
            self.rec.record(self.t, tau=self.last_tau, tau_limit=self.body_limit, band_enabled=band is not None)
        if token is None:
            action = np.zeros(29)
        else:
            obs = self.build_obs(token)
            self.last_obs = obs
            action = self.infer(obs)
        self.last_action = action
        self.targets = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        for _ in range(DECIMATION):
            self._physics_step(band)
        self.tick += 1
        return action

    def _physics_step(self, band=None):
        m, d = self.m, self.d
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        tau = KPS * (self.targets - q) - KDS * dq
        tau = np.clip(tau, -self.body_limit, self.body_limit)
        d.ctrl[:] = 0.0
        d.ctrl[self.body_act] = tau
        if len(self.hand_act):
            ht = -HAND_KP * d.qpos[self.hand_qadr] - HAND_KD * d.qvel[self.hand_vadr]
            d.ctrl[self.hand_act] = np.clip(ht, -self.act_limit[self.hand_act], self.act_limit[self.hand_act])
        if band is not None:
            band(m, d)
        else:
            d.xfrc_applied[:] = 0.0
        mujoco.mj_step(m, d)
        self.last_tau = tau
        self.t += SIM_DT


def elastic_band(point=(0.0, 0.0, 1.0), length=0.0):
    """Same spring as unitree_sdk2py_bridge.ElasticBand, attached to the pelvis."""
    from scipy.spatial.transform import Rotation

    point = np.asarray(point)

    def apply(m, d):
        b = m.body("pelvis").id
        pos, quat = d.xpos[b], d.xquat[b]
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        ang, lin = vel[:3], vel[3:]
        f = 10000 * (point - pos + np.array([0, 0, length])) + 1000 * (0 - lin)
        rotvec = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_rotvec()
        tq = -1000 * rotvec - 10 * ang
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[b] = np.concatenate([f, tq])

    return apply
