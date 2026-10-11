"""Per-tick physical state recording and trial summaries for MuJoCo G1 runs.

Used in two places:
  * ``EvalRecorder`` is attached to the interactive sim (base_sim.py, opt-in via
    ``EVAL_LOG_DIR``) so C++-driven runs get root pose / contact / fall logs, which the
    deploy CSVs cannot provide (LowState has no root position).
  * ``decoder_probe.py`` uses the same recorder for its batch trials.

A "tick" is one 50 Hz control period. Everything is stored as numpy arrays and written
to ``eval_log.npz`` plus a human-readable ``summary.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

try:
    from .common import tilt_deg, yaw_of
except ImportError:
    from common import tilt_deg, yaw_of

# Body groups used for contact classification.
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEE_BODIES = ("left_knee_link", "right_knee_link")
TRUNK_BODIES = ("pelvis", "torso_link", "waist_yaw_link", "waist_roll_link")
FALL_HEIGHT = 0.25  # pelvis height below which we call it a fall (sim auto-resets at 0.2)
FALL_TILT_DEG = 75.0  # torso tilt above which we call it a fall


def _subtree_bodies(model, root_id):
    out = []
    for b in range(model.nbody):
        p = b
        while p != 0 and p != root_id:
            p = model.body_parentid[p]
        if p == root_id or b == root_id:
            out.append(b)
    return out


class EvalRecorder:
    """Records robot state once per control tick from a live MjModel/MjData."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body_qpos_adr=None, body_dof_adr=None):
        self.m = model
        self.d = data
        self.pelvis = model.body("pelvis").id
        self.torso = model.body("torso_link").id
        self.feet = [model.body(n).id for n in FOOT_BODIES]
        robot_bodies = _subtree_bodies(model, self.pelvis)
        self.robot_body_mask = np.zeros(model.nbody, bool)
        self.robot_body_mask[robot_bodies] = True
        self.floor_geoms = {i for i in range(model.ngeom) if model.geom(i).name == "floor"}
        self.obstacle_geoms = {i for i in range(model.ngeom) if model.geom(i).name.startswith("obs_")}
        self.obstacle_names = sorted(model.geom(i).name for i in self.obstacle_geoms)
        self.body_groups = {
            "foot": {model.body(n).id for n in FOOT_BODIES},
            "knee": {model.body(n).id for n in KNEE_BODIES},
            "trunk": {model.body(n).id for n in TRUNK_BODIES},
        }
        self.body_qpos_adr = body_qpos_adr
        self.body_dof_adr = body_dof_adr
        self.events: list[dict] = []
        self.rows: dict[str, list] = {k: [] for k in (
            "t", "wall", "root_pos", "root_quat", "root_linvel", "torso_quat", "pelvis_z", "tilt", "torso_tilt",
            "yaw", "com", "foot_pos", "foot_contact", "floor_contact_nonfoot", "trunk_floor",
            "knee_floor", "obstacle_contact", "obstacle_contact_trunk", "self_collision",
            "q", "dq", "tau", "tau_sat", "band",
        )}

    # ------------------------------------------------------------------
    def _contacts(self):
        m, d = self.m, self.d
        foot = [False, False]
        nonfoot_floor = trunk_floor = knee_floor = obstacle = obstacle_trunk = selfcol = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            r1, r2 = self.robot_body_mask[b1], self.robot_body_mask[b2]
            if r1 and r2:
                selfcol = True
                continue
            if not (r1 or r2):
                continue
            rb, other = (b1, g2) if r1 else (b2, g1)
            if other in self.floor_geoms:
                if rb == self.feet[0]:
                    foot[0] = True
                elif rb == self.feet[1]:
                    foot[1] = True
                else:
                    nonfoot_floor = True
                    if rb in self.body_groups["trunk"]:
                        trunk_floor = True
                    if rb in self.body_groups["knee"]:
                        knee_floor = True
            elif other in self.obstacle_geoms:
                obstacle = True
                if rb in self.body_groups["trunk"]:
                    obstacle_trunk = True
        return foot, nonfoot_floor, trunk_floor, knee_floor, obstacle, obstacle_trunk, selfcol

    def record(self, t: float, tau=None, tau_limit=None, band_enabled: bool = False):
        m, d = self.m, self.d
        quat = d.qpos[3:7].copy()
        torso_quat = d.xquat[self.torso].copy()
        foot, nonfoot, trunk, knee, obst, obst_trunk, selfcol = self._contacts()
        r = self.rows
        r["t"].append(t)
        r["wall"].append(time.time())
        r["root_pos"].append(d.qpos[:3].copy())
        r["root_quat"].append(quat)
        r["root_linvel"].append(d.qvel[:3].copy())
        r["torso_quat"].append(torso_quat)
        r["pelvis_z"].append(float(d.qpos[2]))
        r["tilt"].append(tilt_deg(quat))
        r["torso_tilt"].append(tilt_deg(torso_quat))
        r["yaw"].append(yaw_of(quat))
        r["com"].append(d.subtree_com[self.pelvis].copy())
        r["foot_pos"].append(np.concatenate([d.xpos[f] for f in self.feet]))
        r["foot_contact"].append(foot)
        r["floor_contact_nonfoot"].append(nonfoot)
        r["trunk_floor"].append(trunk)
        r["knee_floor"].append(knee)
        r["obstacle_contact"].append(obst)
        r["obstacle_contact_trunk"].append(obst_trunk)
        r["self_collision"].append(selfcol)
        if self.body_qpos_adr is not None:
            r["q"].append(d.qpos[self.body_qpos_adr].copy())
            r["dq"].append(d.qvel[self.body_dof_adr].copy())
        if tau is not None:
            tau = np.asarray(tau)
            r["tau"].append(tau.copy())
            if tau_limit is not None:
                r["tau_sat"].append(float(np.mean(np.abs(tau) >= 0.99 * np.asarray(tau_limit))))
        r["band"].append(band_enabled)

    def add_event(self, t: float, kind: str, **info):
        self.events.append({"t": float(t), "wall": time.time(), "kind": kind, **info})

    # ------------------------------------------------------------------
    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.rows.items() if len(v)}

    def save(self, out_dir: Path, extra_summary: dict | None = None, t_start: float | None = None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = self.arrays()
        np.savez_compressed(out_dir / "eval_log.npz", **arr)
        summary = summarize(arr, t_start=t_start)
        summary["events"] = self.events
        summary["obstacles"] = self.obstacle_names
        if extra_summary:
            summary.update(extra_summary)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        return summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def fall_mask(arr: dict) -> np.ndarray:
    """Per-tick fall indicator: trunk on the floor, pelvis very low, or torso tipped over."""
    return (
        np.asarray(arr["trunk_floor"], bool)
        | (np.asarray(arr["pelvis_z"]) < FALL_HEIGHT)
        | (np.asarray(arr["torso_tilt"]) > FALL_TILT_DEG)
    )


def summarize(arr: dict, t_start: float | None = None, settle_window_s: float = 2.0) -> dict:
    """Scalar metrics for a run, computed from ``t_start`` (e.g. band release) onward."""
    if "t" not in arr or len(arr["t"]) == 0:
        return {"n_ticks": 0}
    t = np.asarray(arr["t"])
    sel = np.ones_like(t, bool) if t_start is None else t >= t_start
    if "band" in arr and t_start is None and np.any(arr["band"]):
        # Default: only evaluate after the elastic band was last released.
        on = np.where(np.asarray(arr["band"], bool))[0]
        sel = np.arange(len(t)) > on[-1]
    if not np.any(sel):
        return {"n_ticks": int(len(t)), "evaluated_ticks": 0}
    a = {k: np.asarray(v)[sel] for k, v in arr.items() if len(v) == len(t)}
    tt = a["t"]
    falls = fall_mask(a)
    fell = bool(np.any(falls))
    t_fall = float(tt[np.argmax(falls)] - tt[0]) if fell else None
    dur = float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0
    # Stats over the pre-fall portion
    pre = np.arange(len(tt)) < (np.argmax(falls) if fell else len(tt))
    pos = a["root_pos"][pre]
    win = tt[pre] >= (tt[pre][-1] - settle_window_s) if np.any(pre) else pre
    out = {
        "n_ticks": int(len(t)),
        "evaluated_ticks": int(len(tt)),
        "duration_s": dur,
        "fell": fell,
        "time_to_fall_s": t_fall,
        "survival_s": t_fall if fell else dur,
        "pelvis_z_mean": float(np.mean(a["pelvis_z"][pre])) if np.any(pre) else None,
        "pelvis_z_min": float(np.min(a["pelvis_z"][pre])) if np.any(pre) else None,
        "tilt_max_deg": float(np.max(a["torso_tilt"][pre])) if np.any(pre) else None,
        "root_displacement_xy": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])) if len(pos) else None,
        "root_path_length_xy": float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))) if len(pos) > 1 else 0.0,
        "yaw_change_deg": float(np.degrees(np.unwrap(a["yaw"][pre])[-1] - a["yaw"][pre][0])) if np.any(pre) else None,
        "final_root_xy": pos[-1, :2].tolist() if len(pos) else None,
        "max_root_x": float(np.max(pos[:, 0])) if len(pos) else None,
        "frac_nonfoot_floor_contact": float(np.mean(a["floor_contact_nonfoot"][pre])) if np.any(pre) else None,
        "frac_knee_floor_contact": float(np.mean(a["knee_floor"][pre])) if np.any(pre) else None,
        "frac_self_collision": float(np.mean(a["self_collision"][pre])) if np.any(pre) else None,
        "obstacle_contact_ticks": int(np.sum(a["obstacle_contact"])),
        "obstacle_trunk_contact_ticks": int(np.sum(a["obstacle_contact_trunk"])),
    }
    if np.any(win):
        lin = a["root_linvel"][pre][win]
        out["settle_root_speed_xy"] = float(np.mean(np.linalg.norm(lin[:, :2], axis=1)))
        out["settle_pelvis_z_std"] = float(np.std(a["pelvis_z"][pre][win]))
        if "dq" in a and len(a["dq"]) == len(tt):
            out["settle_joint_vel_rms"] = float(np.sqrt(np.mean(a["dq"][pre][win] ** 2)))
        fc = np.asarray(a["foot_contact"][pre][win], bool)
        out["settle_double_support_frac"] = float(np.mean(fc.all(axis=1)))
    if "tau_sat" in a and len(a["tau_sat"]) == len(tt):
        out["torque_saturation_frac"] = float(np.mean(a["tau_sat"][pre])) if np.any(pre) else None
    out["outcome"] = classify(out)
    return out


def classify(s: dict) -> str:
    """Coarse label: fell / static (stands still) / moving (stable but locomoting or fidgeting)."""
    if s.get("fell"):
        return "fell"
    speed = s.get("settle_root_speed_xy")
    jv = s.get("settle_joint_vel_rms")
    if speed is not None and speed < 0.05 and (jv is None or jv < 0.3):
        return "static"
    return "moving"
