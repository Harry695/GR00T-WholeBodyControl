"""Turn investigation runs into figures + a markdown report.

Subcommands:
    analyze.py probe   runs/probe/<dir> [<dir> ...]    batch decoder-probe results (part 2, CPU)
    analyze.py part2   runs/part2/<dir>                 C++ token-streaming run (scores each trial)
    analyze.py part1   runs/part1/<dir> [<dir> ...]    planner scenarios (obstacle success metrics)
    analyze.py library                                  token-library usage (how much of the grid is used)

Outputs go to <dir>/analysis/ (or runs/analysis/ for multi-run / library reports):
report.md, PNG figures, and machine-readable JSON.

Outcome labels per trial (metrics.classify): fell / static (stands still after settling)
/ moving (no fall but walking, drifting or fidgeting).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

try:
    from .build_token_library import library_tokens, load_library
    from .common import RUNS_DIR, TOKEN_DIM, value_to_level
    from .make_scenes import BAR_X, GAP_X, HURDLE_X
    from .metrics import _json_default, summarize
except ImportError:
    from build_token_library import library_tokens, load_library
    from common import RUNS_DIR, TOKEN_DIM, value_to_level
    from make_scenes import BAR_X, GAP_X, HURDLE_X
    from metrics import _json_default, summarize

# --- style (validated: categorical slots pass CVD checks for all pairs) ------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
OUTCOME_COLORS = {"static": "#2a78d6", "moving": "#eda100", "fell": "#e34948"}
OUTCOMES = ["static", "moving", "fell"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "axes.axisbelow": True,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _load_results(dirs) -> list[dict]:
    res = []
    for d in dirs:
        f = Path(d) / "results.jsonl"
        if f.exists():
            res += [json.loads(line) for line in open(f)]
    return res


def _rate(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# =========================================================================================
# Part 2: probe results
# =========================================================================================
def outcome_legend(ax):
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=OUTCOME_COLORS[o], label=o) for o in OUTCOMES],
              loc="upper left", bbox_to_anchor=(1.01, 1.0))


def plot_dim_sweep(res, out_dir: Path, md: list):
    by_base = defaultdict(list)
    for r in res:
        by_base[r["meta"]["base"]].append(r)
    for base, rs in by_base.items():
        grid = np.full((TOKEN_DIM, 32), np.nan)
        surv = np.full((TOKEN_DIM, 32), np.nan)
        base_lv = {}
        for r in rs:
            d, lv = r["meta"]["dim"], r["meta"]["level"]
            grid[d, lv + 16] = OUTCOMES.index(r["summary"]["outcome"])
            surv[d, lv + 16] = r["summary"]["survival_s"]
            base_lv[d] = r["meta"]["base_level"]
        dims = sorted(base_lv)
        safe = f"{base.replace(':', '_').replace('/', '_')}"
        fig, ax = plt.subplots(figsize=(9, max(3, 0.13 * len(dims) + 1.2)))
        ax.imshow(grid[dims], aspect="auto", cmap=ListedColormap([OUTCOME_COLORS[o] for o in OUTCOMES]),
                  vmin=-0.5, vmax=2.5, interpolation="nearest", extent=(-16.5, 15.5, len(dims) - 0.5, -0.5))
        ax.scatter([base_lv[d] for d in dims], range(len(dims)), marker="o", s=10, facecolor=SURFACE,
                   edgecolor=INK, linewidth=0.8, label="base token level", zorder=3)
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims if len(dims) <= 64 else [], fontsize=6)
        ax.set_xlabel("level of the swept dim (value = level / 16)")
        ax.set_ylabel("token dim")
        ax.set_title(f"Outcome when one dim of '{base}' is set to each level (others fixed)")
        ax.grid(False)
        outcome_legend(ax)
        f1 = _save(fig, out_dir / f"dim_sweep_outcome_{safe}.png")

        # fall rate vs distance from the base level
        deltas = np.array([r["meta"]["delta"] for r in rs])
        fell = np.array([r["summary"]["fell"] for r in rs])
        moving = np.array([r["summary"]["outcome"] == "moving" for r in rs])
        uniq = np.arange(-31, 32)
        fr = [(_rate(fell[deltas == u]) if np.any(deltas == u) else np.nan) for u in uniq]
        mr = [(_rate(moving[deltas == u]) if np.any(deltas == u) else np.nan) for u in uniq]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(uniq, fr, color=OUTCOME_COLORS["fell"], label="fell")
        ax.plot(uniq, mr, color=OUTCOME_COLORS["moving"], label="moving")
        ax.set_xlabel("level offset from base on a single dim")
        ax.set_ylabel("fraction of dims")
        ax.set_ylim(0, 1)
        ax.set_title(f"Single-dim edits of '{base}': outcome vs offset")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        f2 = _save(fig, out_dir / f"dim_sweep_offset_{safe}.png")

        # per-dim tolerance: contiguous safe interval around base
        tol = {}
        for d in dims:
            b = base_lv[d] + 16
            ok = grid[d] != 2
            lo = b
            while lo - 1 >= 0 and ok[lo - 1]:
                lo -= 1
            hi = b
            while hi + 1 < 32 and ok[hi + 1]:
                hi += 1
            tol[d] = (lo - 16, hi - 16)
        widths = np.array([tol[d][1] - tol[d][0] + 1 for d in dims])
        md += [f"### dim_sweep, base `{base}`",
               f"- trials: {len(rs)}, fell: {fell.mean():.1%}, moving: {moving.mean():.1%}, "
               f"static: {1 - fell.mean() - moving.mean():.1%}",
               f"- dims where **every** level is non-falling: {int(np.sum(widths == 32))}/{len(dims)}; "
               f"median contiguous non-falling interval around base: {np.median(widths):.0f} of 32 levels",
               f"- most fragile dims (narrowest safe interval): "
               + ", ".join(f"{d} [{tol[d][0]},{tol[d][1]}]" for d in np.array(dims)[np.argsort(widths)[:8]]),
               f"![]({f1})", f"![]({f2})", ""]
        json.dump({"base": base, "tolerance": tol}, open(out_dir / f"dim_sweep_tolerance_{safe}.json", "w"),
                  default=_json_default, indent=1)


def plot_perturb(res, out_dir: Path, md: list):
    by_base = defaultdict(list)
    for r in res:
        by_base[r["meta"]["base"]].append(r)
    fig, axes = plt.subplots(1, len(by_base), figsize=(4.2 * len(by_base) + 1.5, 3.2), squeeze=False)
    rows = []
    for ax, (base, rs) in zip(axes[0], by_base.items()):
        nd = sorted({r["meta"]["ndims"] for r in rs})
        rad = sorted({r["meta"]["radius"] for r in rs})
        for i, m in enumerate(nd[:4]):
            ys = [_rate([r["summary"]["fell"] for r in rs if r["meta"]["ndims"] == m and r["meta"]["radius"] == k])
                  for k in rad]
            ax.plot(rad, ys, color=SERIES[i], marker="o", markersize=4, label=f"{m} dims")
            for k, y in zip(rad, ys):
                rows.append([base, m, k, f"{y:.0%}"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(rad)
        ax.set_xticklabels(rad)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("offset per perturbed dim (levels)")
        ax.set_ylabel("fall rate")
        ax.set_title(f"base '{base}'", fontsize=9)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    f = _save(fig, out_dir / "perturb_fall_rate.png")
    md += ["### perturb: random +-k level offsets on m dims", f"![]({f})", "",
           _md_table(["base", "dims perturbed", "offset", "fall rate"], rows), ""]


def plot_interp(res, out_dir: Path, md: list):
    alphas = sorted({r["meta"]["alpha"] for r in res})
    ys = [_rate([r["summary"]["fell"] for r in res if r["meta"]["alpha"] == a]) for a in alphas]
    ms = [_rate([r["summary"]["outcome"] == "moving" for r in res if r["meta"]["alpha"] == a]) for a in alphas]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(alphas, ys, color=OUTCOME_COLORS["fell"], marker="o", markersize=4, label="fell")
    ax.plot(alphas, ms, color=OUTCOME_COLORS["moving"], marker="o", markersize=4, label="moving")
    ax.set_xlabel("interpolation alpha (0 = token A, 1 = token B)")
    ax.set_ylabel("fraction of pairs")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Grid-snapped interpolation between two real tokens")
    f = _save(fig, out_dir / "interp.png")
    md += ["### interp", f"![]({f})", ""]


def plot_survival(res_by_set: dict, out_dir: Path, md: list):
    sets = [s for s in ("random_hold", "random_marginal", "random_resample", "in_dist_hold") if s in res_by_set]
    if not sets:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(0, max(r["schedule_s"] for s in sets for r in res_by_set[s]) + 0.01, 25)
    for i, s in enumerate(sets):
        sv = [r["summary"]["survival_s"] for r in res_by_set[s]]
        ax.hist(sv, bins=bins, histtype="step", color=SERIES[i], linewidth=2, label=s, density=True)
    ax.set_xlabel("time until fall (s); right edge = never fell")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("How long the decoder keeps the robot up under a held token")
    f = _save(fig, out_dir / "survival.png")
    md += ["### survival under held tokens", f"![]({f})", ""]


def distance_to_library(res, out_dir: Path, md: list):
    lib = load_library()
    L = value_to_level(library_tokens(lib))
    if len(L) == 0:
        return
    if len(L) > 6000:
        L = L[np.linspace(0, len(L) - 1, 6000).astype(int)]
    pts = [(r, value_to_level(np.asarray(r["probe_token"]))) for r in res if r.get("probe_token") is not None]
    if not pts:
        return
    P = np.stack([p for _, p in pts])
    dist = np.array([np.abs(L - p).sum(1).min() for p in P])
    fell = np.array([r["summary"]["fell"] for r, _ in pts])
    moving = np.array([r["summary"]["outcome"] == "moving" for r, _ in pts])
    edges = np.unique(np.quantile(dist, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        edges = np.array([dist.min(), dist.max() + 1])
    idx = np.clip(np.digitize(dist, edges[1:-1]), 0, len(edges) - 2)
    centers, fr, mr, ns = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not np.any(sel):
            continue
        centers.append(f"{edges[b]:.0f}-{edges[b + 1]:.0f}")
        fr.append(fell[sel].mean())
        mr.append(moving[sel].mean())
        ns.append(int(sel.sum()))
    x = np.arange(len(centers))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x, fr, width=0.8, color=OUTCOME_COLORS["fell"], label="fell")
    ax.bar(x, mr, width=0.8, bottom=fr, color=OUTCOME_COLORS["moving"], label="moving",
           edgecolor=SURFACE, linewidth=1)
    for xi, n in zip(x, ns):
        ax.text(xi, 1.02, f"n={n}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(centers, rotation=30, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("L1 distance (levels) from the nearest real token in the library")
    ax.set_ylabel("fraction of trials")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Stability vs distance from the data manifold (all probe sets)")
    f = _save(fig, out_dir / "distance_to_library.png")
    md += ["### Stability vs distance from real tokens",
           "Distance = sum over 64 dims of |level difference| to the closest token encoded from real motion. "
           "Uniformly random tokens sit at ~600; real tokens in the library are 0.", f"![]({f})", ""]


def analyze_probe(dirs, out_dir: Path | None = None, results=None):
    res = results if results is not None else _load_results(dirs)
    if not res:
        raise SystemExit("no results.jsonl found")
    out_dir = out_dir or (Path(dirs[0]) / "analysis" if len(dirs) == 1 else RUNS_DIR / "analysis" / "probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_set = defaultdict(list)
    for r in res:
        by_set[r["set"]].append(r)
    md = ["# Decoder probe report", "", f"Runs: {', '.join(str(d) for d in dirs)}", ""]
    rows = []
    for s, rs in by_set.items():
        c = Counter(r["summary"]["outcome"] for r in rs)
        rec = [r["recovery"] for r in rs if r.get("recovery")]
        rec_ok = _rate([not x["fell"] and x["outcome"] == "static" for x in rec]) if rec else float("nan")
        rows.append([s, len(rs), f"{c['static'] / len(rs):.0%}", f"{c['moving'] / len(rs):.0%}",
                     f"{c['fell'] / len(rs):.0%}", f"{np.median([r['summary']['survival_s'] for r in rs]):.1f}",
                     f"{rec_ok:.0%}" if rec else "-"])
    md += ["## Outcomes by set",
           _md_table(["set", "trials", "static", "moving", "fell", "median survival (s)",
                      "recovered to static stand (non-fallen)"], rows), "",
           "- **static**: stands still at the end of the hold; **moving**: no fall but locomoting/drifting/"
           "fidgeting; **fell**: trunk touched the floor, pelvis < 0.25 m, or torso tilt > 75 deg.", ""]
    md.append("## Figures")
    if "dim_sweep" in by_set:
        plot_dim_sweep(by_set["dim_sweep"], out_dir, md)
    if "perturb" in by_set:
        plot_perturb(by_set["perturb"], out_dir, md)
    if "interp" in by_set:
        plot_interp(by_set["interp"], out_dir, md)
    plot_survival(by_set, out_dir, md)
    distance_to_library(res, out_dir, md)
    if "in_dist_replay" in by_set:
        md += ["### in_dist_replay (positive control: real token sequences replayed open-loop)",
               _md_table(["sequence", "outcome", "survival (s)", "displacement (m)"],
                         [[r["meta"]["seq"], r["summary"]["outcome"], f"{r['summary']['survival_s']:.1f}",
                           f"{r['summary'].get('root_displacement_xy') or 0:.2f}"] for r in by_set["in_dist_replay"]]),
               ""]
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"[analyze] wrote {out_dir / 'report.md'}")
    return out_dir


# =========================================================================================
# Part 2 via the C++ stack: slice the sim eval log by the streamer's timeline
# =========================================================================================
def analyze_part2(run_dir: Path):
    run_dir = Path(run_dir)
    arr = dict(np.load(run_dir / "sim" / "eval_log.npz"))
    trials = {t["id"]: t for t in map(json.loads, open(run_dir / "trials.jsonl"))}
    tl = [json.loads(x) for x in open(run_dir / "timeline.jsonl")]
    begins = {e["id"]: e["wall"] for e in tl if e["kind"] == "trial_begin"}
    ends = {e["id"]: e["wall"] for e in tl if e["kind"] == "trial_end"}
    wall = arr["wall"]
    results = []
    for tid, t0 in begins.items():
        t1 = ends.get(tid)
        if t1 is None:
            continue
        sel = (wall >= t0) & (wall <= t1)
        if sel.sum() < 5:
            continue
        sub = {k: v[sel] for k, v in arr.items() if len(v) == len(wall)}
        s = summarize(sub, t_start=sub["t"][0])
        tr = trials[tid]
        results.append({"id": tid, "set": tr["set"], "meta": tr["meta"], "probe_token": tr["probe_token"],
                        "summary": s, "recovery": None, "schedule_s": float(t1 - t0)})
    with open(run_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=_json_default) + "\n")
    print(f"[analyze] scored {len(results)} trials from the C++ run")
    analyze_probe([run_dir], out_dir=run_dir / "analysis", results=results)


# =========================================================================================
# Part 1: planner scenarios
# =========================================================================================
def _obstacle_x(scene: str):
    if scene.startswith("hurdle"):
        return HURDLE_X
    if scene.startswith("gap"):
        return GAP_X
    if scene.startswith("lowbar"):
        return BAR_X
    return None


def analyze_part1_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    arr = dict(np.load(run_dir / "sim" / "eval_log.npz"))
    sim_summary = json.load(open(run_dir / "sim" / "summary.json"))
    scen = json.load(open(run_dir / "scenario.json"))
    tl = [json.loads(x) for x in open(run_dir / "timeline.jsonl")]
    scene = Path(sim_summary.get("scene", "")).stem.replace("scene_inv_", "").replace("scene_43dof", "flat")
    w0 = next((e["wall"] for e in tl if e["kind"] == "scenario_begin"), None)
    w1 = next((e["wall"] for e in tl if e["kind"] == "scenario_end"), None)
    if w0 is None:
        raise SystemExit(f"{run_dir}: scenario never started (no scenario_begin in timeline)")
    w1 = (w1 or arr["wall"][-1]) + 3.0
    wall = arr["wall"]
    sel = (wall >= w0) & (wall <= w1)
    sub = {k: v[sel] for k, v in arr.items() if len(v) == len(wall)}
    s = summarize(sub, t_start=sub["t"][0])
    ox = _obstacle_x(scene)
    pos = sub["root_pos"]
    falls = s["fell"]
    if ox is not None:
        s["obstacle_x"] = ox
        s["passed_obstacle"] = bool(np.max(pos[:, 0]) > ox + 0.4) and not falls
        crossed = np.where(pos[:, 0] > ox)[0]
        s["time_to_cross_s"] = float(sub["t"][crossed[0]] - sub["t"][0]) if len(crossed) else None
    s["scene"] = scene
    s["scenario"] = scen["name"]
    s["sim_events"] = [e for e in sim_summary.get("events", []) if w0 <= e.get("wall", 0) <= w1]

    # --- figure: x/z over time with step boundaries, and top-down path -------------------
    t = sub["t"] - sub["t"][0]
    steps = [e for e in tl if e["kind"] == "step"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    ax = axes[0]
    ax.plot(t, pos[:, 0], color=SERIES[0], label="pelvis x")
    ax.plot(t, sub["pelvis_z"], color=SERIES[1], label="pelvis z")
    if ox is not None:
        ax.axhline(ox, color=MUTED, linewidth=1, linestyle="--")
        ax.text(t[-1], ox, " obstacle x", va="bottom", ha="right", fontsize=7, color=MUTED)
    for e in steps:
        ts = e["wall"] - w0
        ax.axvline(ts, color=GRID, linewidth=1)
        ax.text(ts, ax.get_ylim()[1], f" {e['mode']}", rotation=90, va="top", fontsize=6, color=MUTED)
    oc = np.asarray(sub["obstacle_contact"], bool)
    if oc.any():
        ax.scatter(t[oc], pos[oc, 0], s=8, color=SERIES[3], label="obstacle contact", zorder=3)
    ax.set_xlabel("time since scenario start (s)")
    ax.set_ylabel("m")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(f"{scen['name']} on {scene}")
    ax = axes[1]
    ax.plot(t, sub["torso_tilt"], color=SERIES[2])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("torso tilt (deg)")
    ax.set_title("torso tilt")
    ax = axes[2]
    ax.plot(pos[:, 0], pos[:, 1], color=SERIES[0])
    fp = sub["foot_pos"]
    ax.scatter(fp[::5, 0], fp[::5, 1], s=3, color=MUTED, label="left foot")
    ax.scatter(fp[::5, 3], fp[::5, 4], s=3, color=INK2, label="right foot")
    if ox is not None:
        ax.axvline(ox, color=SERIES[3], linewidth=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("top-down path (pelvis line, feet dots)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    out = run_dir / "analysis"
    s["figure"] = _save(fig, out / "trajectory.png")
    (out / "metrics.json").write_text(json.dumps(s, indent=2, default=_json_default))
    return s


def analyze_part1(dirs):
    rows, all_s = [], []
    for d in dirs:
        try:
            s = analyze_part1_run(Path(d))
        except Exception as e:  # keep going over many runs
            print(f"[analyze] {d}: {e}")
            continue
        all_s.append((d, s))
        rows.append([s["scenario"], s["scene"], s["outcome"], s.get("passed_obstacle", "-"),
                     f"{s.get('time_to_cross_s') or float('nan'):.1f}", f"{s['max_root_x'] or 0:.2f}",
                     s["obstacle_contact_ticks"], s["obstacle_trunk_contact_ticks"],
                     f"{s['frac_knee_floor_contact'] or 0:.0%}", f"{s['tilt_max_deg'] or 0:.0f}"])
    md = ["# Part 1: planner-command scenarios", "",
          _md_table(["scenario", "scene", "outcome", "passed obstacle", "t cross (s)", "max x (m)",
                     "obstacle contact ticks", "trunk-obstacle ticks", "knee-floor", "max tilt"], rows), "",
          "- *passed obstacle*: pelvis got 0.4 m past the obstacle plane without falling.",
          "- contact ticks are 50 Hz samples where any robot geom touched an `obs_*` geom (trunk = pelvis/torso).",
          ""]
    for d, s in all_s:
        md += [f"## {s['scenario']} / {s['scene']}", f"![]({Path(d).resolve() / 'analysis' / s['figure']})", ""]
    out = Path(dirs[0]) / "analysis" if len(dirs) == 1 else RUNS_DIR / "analysis" / "part1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md))
    print(f"[analyze] wrote {out / 'report.md'}")


# =========================================================================================
# Token library usage
# =========================================================================================
def analyze_library():
    lib = load_library()
    T = library_tokens(lib)
    if len(T) == 0:
        raise SystemExit("token library is empty; run build_token_library.py")
    L = value_to_level(T)
    hist = np.stack([np.bincount(L[:, d] + 16, minlength=32) for d in range(TOKEN_DIM)]).astype(float)
    hist /= hist.sum(1, keepdims=True)
    out = RUNS_DIR / "analysis" / "library"
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(hist, aspect="auto", cmap=BLUES, extent=(-16.5, 15.5, 63.5, -0.5), interpolation="nearest")
    ax.set_xlabel("level")
    ax.set_ylabel("token dim (0-31 = token 1, 32-63 = token 2)")
    ax.set_title(f"How often each level is used by real motion ({len(T)} tokens, {len(lib)} sequences)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="fraction of tokens")
    f = _save(fig, out / "level_usage.png")
    used = (hist > 0).sum(1)
    q = np.quantile(L, [0.01, 0.99], axis=0)
    md = ["# Token library", "", f"- {len(lib)} sequences, {len(T)} tokens; sources: "
          + ", ".join(sorted({k.split(':')[0] for k in lib})),
          f"- levels ever used per dim: min {used.min()}, median {np.median(used):.0f}, max {used.max()} (of 32)",
          f"- per-dim std: median {np.median(L.std(0)):.2f} levels (uniform random would be 9.2)",
          f"- 98% of values per dim lie within levels [{np.median(q[0]):.0f}, {np.median(q[1]):.0f}] (median over dims)",
          f"![]({f})", ""]
    (out / "report.md").write_text("\n".join(md))
    print(f"[analyze] wrote {out / 'report.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("dirs", nargs="+")
    p = sub.add_parser("part2")
    p.add_argument("dir")
    p = sub.add_parser("part1")
    p.add_argument("dirs", nargs="+")
    sub.add_parser("library")
    args = ap.parse_args()
    if args.cmd == "probe":
        analyze_probe(args.dirs)
    elif args.cmd == "part2":
        analyze_part2(Path(args.dir))
    elif args.cmd == "part1":
        analyze_part1(args.dirs)
    elif args.cmd == "library":
        analyze_library()


if __name__ == "__main__":
    main()
