"""Hand-constructed token schedules for probing the SONIC control decoder.

A *trial* is a dict (JSON-serialisable) with:
    id, set, meta (free-form labels used by analyze.py), probe_token (64 floats or None),
    segments: list of {"token": [64 floats], "dur": seconds}
              or {"seq": <library name>, "start": i, "end": j}  (one token per 50 Hz tick)

Every trial is preceded by a standing phase (standing token) handled by the runner, so the
robot always starts from the same stable state. Tokens are always on the FSQ grid (k/16).

Sets (``--set`` in decoder_probe.py / stream_tokens.py):
  sanity          standing / zero tokens held
  random_hold     i.i.d. uniform levels per dim, held           (how much of the full cube is OK?)
  random_resample i.i.d. uniform tokens switched every --resample-s
  random_marginal each dim sampled from its empirical in-distribution marginal (independent dims)
  dim_sweep       base token with ONE dim set to each of the 32 levels, held
  perturb         base + random +-k level offsets on m random dims (distance-from-data curve)
  interp          grid-snapped linear interpolation between two library tokens
  in_dist_hold    real library tokens frozen and held                   (static vs. dynamic meaning)
  in_dist_replay  real library token sequences replayed open-loop      (positive control)
"""

from __future__ import annotations

import numpy as np

try:
    from .build_token_library import library_tokens, load_library
    from .common import LEVEL_INDICES, STANDING_TOKEN, TOKEN_DIM, level_to_value, value_to_level
except ImportError:
    from build_token_library import library_tokens, load_library
    from common import LEVEL_INDICES, STANDING_TOKEN, TOKEN_DIM, level_to_value, value_to_level

SETS = ["sanity", "random_hold", "random_resample", "random_marginal", "dim_sweep", "perturb", "interp",
        "in_dist_hold", "in_dist_replay"]


def _tok(v):
    return [float(x) for x in np.asarray(v).reshape(TOKEN_DIM)]


def resolve_base(name: str, lib: dict) -> np.ndarray:
    """'standing' | 'zero' | '<library seq>' (its medoid) | '<library seq>@<frame>'."""
    if name == "standing":
        return STANDING_TOKEN.copy()
    if name == "zero":
        return np.zeros(TOKEN_DIM, np.float32)
    seq, _, frame = name.partition("@")
    if seq not in lib:
        raise KeyError(f"base '{seq}' not in token library (have: {sorted(lib)[:8]}...)")
    toks = lib[seq]
    if frame:
        return toks[int(frame)].copy()
    return medoid(toks)


def medoid(toks: np.ndarray) -> np.ndarray:
    L = value_to_level(toks)
    if len(L) > 600:
        L = L[np.linspace(0, len(L) - 1, 600).astype(int)]
    dist = np.abs(L[:, None, :] - L[None, :, :]).sum(-1)
    return level_to_value(L[np.argmin(dist.sum(1))])


def build_trials(set_name: str, n: int = 100, hold_s: float = 6.0, seed: int = 0, bases=("standing",),
                 dims=None, levels=None, radii=(1, 2, 4, 8), ndims=(1, 4, 16, 64), resample_s: float = 1.0,
                 alphas=(0.0, 0.25, 0.5, 0.75, 1.0), library=None) -> list[dict]:
    lib = load_library() if library is None else library
    rng = np.random.default_rng(seed)
    trials: list[dict] = []

    def add(meta, segments, probe):
        trials.append({"id": f"{set_name}_{len(trials):05d}", "set": set_name, "meta": meta,
                       "segments": segments, "probe_token": None if probe is None else _tok(probe)})

    if set_name == "sanity":
        for b in ("standing", "zero"):
            tok = resolve_base(b, lib)
            add({"base": b}, [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "random_hold":
        for i in range(n):
            tok = level_to_value(rng.integers(LEVEL_INDICES[0], LEVEL_INDICES[-1] + 1, TOKEN_DIM))
            add({"seed": seed, "i": i}, [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "random_resample":
        k = max(1, int(round(hold_s / resample_s)))
        for i in range(n):
            segs = [{"token": _tok(level_to_value(rng.integers(-16, 16, TOKEN_DIM))), "dur": resample_s}
                    for _ in range(k)]
            add({"seed": seed, "i": i, "resample_s": resample_s}, segs, None)

    elif set_name == "random_marginal":
        T = value_to_level(library_tokens(lib))
        if len(T) == 0:
            raise RuntimeError("token library is empty; run build_token_library.py first")
        for i in range(n):
            lv = T[rng.integers(0, len(T), TOKEN_DIM), np.arange(TOKEN_DIM)]  # independent per-dim draws
            tok = level_to_value(lv)
            add({"seed": seed, "i": i}, [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "dim_sweep":
        dims = range(TOKEN_DIM) if dims is None else dims
        levels = LEVEL_INDICES if levels is None else levels
        for b in bases:
            base = resolve_base(b, lib)
            base_lv = value_to_level(base)
            for d in dims:
                for lv in levels:
                    tok = base.copy()
                    tok[d] = level_to_value(lv)
                    add({"base": b, "dim": int(d), "level": int(lv), "base_level": int(base_lv[d]),
                         "delta": int(lv - base_lv[d])}, [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "perturb":
        reps = max(1, n)
        for b in bases:
            base_lv = value_to_level(resolve_base(b, lib))
            for r in radii:
                for m in ndims:
                    for i in range(reps):
                        idx = rng.choice(TOKEN_DIM, size=min(m, TOKEN_DIM), replace=False)
                        lv = base_lv.copy()
                        lv[idx] += rng.choice([-r, r], size=len(idx))
                        lv = np.clip(lv, -16, 15)
                        tok = level_to_value(lv)
                        add({"base": b, "radius": int(r), "ndims": int(m), "i": i,
                             "l1": int(np.abs(lv - base_lv).sum())}, [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "interp":
        names = [k for k in lib if k.startswith(("ref_closed:", "deploy:"))] or list(lib)
        for i in range(n):
            a_name, b_name = rng.choice(names, 2, replace=False)
            a = lib[a_name][rng.integers(len(lib[a_name]))]
            b = lib[b_name][rng.integers(len(lib[b_name]))]
            for al in alphas:
                tok = level_to_value(value_to_level((1 - al) * a + al * b))
                add({"pair": i, "a": a_name, "b": b_name, "alpha": float(al)},
                    [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "in_dist_hold":
        names = [k for k in lib if k.startswith(("ref_closed:", "deploy:"))] or list(lib)
        for i in range(n):
            nm = names[rng.integers(len(names))]
            f = int(rng.integers(len(lib[nm])))
            tok = lib[nm][f]
            add({"seq": nm, "frame": f}, [{"token": _tok(tok), "dur": hold_s}], tok)

    elif set_name == "in_dist_replay":
        for nm in sorted(lib):
            if nm.startswith(("ref_closed:", "ref_open:", "deploy:")):
                add({"seq": nm}, [{"seq": nm, "start": 0, "end": len(lib[nm])}], None)

    else:
        raise ValueError(f"unknown set '{set_name}', choose from {SETS}")
    return trials


def trial_schedule(trial: dict, lib: dict, dt: float = 0.02) -> np.ndarray:
    """Expand a trial's segments into a per-tick [T, 64] token array."""
    rows = []
    for seg in trial["segments"]:
        if "token" in seg:
            k = max(1, int(round(seg["dur"] / dt)))
            rows.append(np.repeat(np.asarray(seg["token"], np.float32)[None], k, axis=0))
        else:
            rows.append(lib[seg["seq"]][seg.get("start", 0):seg.get("end")].astype(np.float32))
    return np.concatenate(rows) if rows else np.zeros((0, TOKEN_DIM), np.float32)
