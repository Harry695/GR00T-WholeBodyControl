"""Build the in-distribution token library used as bases, controls and distance references.

Sources:
  1. ``ref_open:<motion>``   encoder applied to each shipped reference motion assuming perfect
                             tracking (base orientation = reference root orientation).
  2. ``ref_closed:<motion>`` tokens actually fed to the decoder while the Python closed loop
                             tracked that motion (encoder sees the real robot orientation).
  3. ``deploy:<run>``        token_state.csv logged by the C++ deploy (``--enable-csv-logs``),
                             e.g. from part-1 planner scenarios. Pass ``--deploy-logs DIR ...``.

Output: ``token_library.npz`` mapping sequence name -> [T, 64] float32 tokens.

Usage:
    python build_token_library.py                         # reference motions only (CPU, ~1 min)
    python build_token_library.py --deploy-logs runs/part1/*/deploy_logs
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

try:
    from .common import INVESTIGATION_DIR, STANDING_TOKEN, TOKEN_DIM
    from .encoder_utils import G1Encoder, delta_heading_for, list_reference_motions, load_reference_motion
except ImportError:
    from common import INVESTIGATION_DIR, STANDING_TOKEN, TOKEN_DIM
    from encoder_utils import G1Encoder, delta_heading_for, list_reference_motions, load_reference_motion

DEFAULT_LIBRARY = INVESTIGATION_DIR / "token_library.npz"


def load_library(path: Path | str = DEFAULT_LIBRARY) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def library_tokens(lib: dict[str, np.ndarray], prefix: str | None = None) -> np.ndarray:
    """All tokens in the library (optionally only sequences whose name starts with prefix)."""
    seqs = [v for k, v in lib.items() if prefix is None or k.startswith(prefix)]
    return np.concatenate(seqs) if seqs else np.zeros((0, TOKEN_DIM), np.float32)


def read_deploy_tokens(log_dir: Path) -> np.ndarray | None:
    """token_state.csv columns: index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp,token_0.."""
    f = Path(log_dir) / "token_state.csv"
    if not f.exists():
        return None
    raw = np.genfromtxt(f, delimiter=",", skip_header=1)
    if raw.ndim == 1:
        raw = raw[None]
    tok = raw[:, 5:5 + TOKEN_DIM]
    tok = tok[np.any(tok != 0, axis=1)]  # drop ticks before the encoder/planner produced tokens
    return tok.astype(np.float32) if len(tok) else None


def encode_reference_motions(lib: dict, closed_loop: bool = True):
    enc = G1Encoder()
    sim = None
    if closed_loop:
        try:
            from .probe_sim import DecoderSim
        except ImportError:
            from probe_sim import DecoderSim
        sim = DecoderSim()
    for p in list_reference_motions():
        mo = load_reference_motion(p)
        lib[f"ref_open:{mo.name}"] = enc.encode_open_loop(mo).astype(np.float32)
        if sim is not None:
            sim.reset()
            for _ in range(50):
                sim.step(STANDING_TOKEN)
            dh = delta_heading_for(sim.d.qpos[3:7], mo, 0)
            toks = []
            for f in range(mo.T):
                tok = enc.encode(mo, f, sim.d.qpos[3:7].copy(), dh)
                toks.append(tok)
                sim.step(tok)
            lib[f"ref_closed:{mo.name}"] = np.asarray(toks, np.float32)
        print(f"encoded {mo.name}: {mo.T} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_LIBRARY))
    ap.add_argument("--deploy-logs", nargs="*", default=[], help="C++ log dirs containing token_state.csv (globs ok)")
    ap.add_argument("--skip-reference", action="store_true", help="keep existing ref_* entries, only add deploy logs")
    ap.add_argument("--open-loop-only", action="store_true", help="skip the closed-loop tracking pass")
    args = ap.parse_args()

    lib = load_library(args.out)
    if not args.skip_reference:
        encode_reference_motions(lib, closed_loop=not args.open_loop_only)
    for pattern in args.deploy_logs:
        for d in sorted(glob.glob(pattern)):
            tok = read_deploy_tokens(Path(d))
            if tok is None:
                print(f"no tokens in {d}")
                continue
            name = f"deploy:{Path(d).resolve().parent.name}/{Path(d).name}"
            lib[name] = tok
            print(f"added {name}: {len(tok)} ticks")
    if STANDING_TOKEN is not None:
        lib["standing"] = STANDING_TOKEN[None].astype(np.float32)
    np.savez_compressed(args.out, **lib)
    n = sum(len(v) for v in lib.values())
    print(f"wrote {args.out}: {len(lib)} sequences, {n} tokens")


if __name__ == "__main__":
    main()
