# SONIC motion-decoder expressivity investigation

Two questions:

1. **Can SONIC's built-in high-level interface make the G1 do what we need?** The behaviours are stepping
   over an obstacle, shimmying through a narrow gap, ducking low and walking under something, and stepping
   slowly through floor clutter with changes of direction.
2. **Does the whole "universal token" space decode to stable behaviour in sim, or only a small region of it?**

## What "the motion decoder" is

SONIC ([arXiv 2511.07820](https://arxiv.org/html/2511.07820v4)) has three encoders, one per kind of
command:

- **robot**: G1 joint trajectories;
- **human**: SMPL;
- **hybrid**: VR head and hands plus lower body.

Each encoder takes the next 1 s of desired motion (10 frames, 0.1 s apart) and compresses it into the
same **universal token**. That token is an FSQ code made of 2 tokens × 32 dims, with 32 levels per dim,
so 64 numbers, each equal to `k/16` for `k ∈ {-16..15}`.

The **control decoder** (`g1_dyn`, file `policy/release/model_decoder.onnx`) is the motion decoder we
probe. It is an MLP:

- **Input (994 values):** the 64-dim token, plus the last 10 ticks (0.2 s) of robot state: angular
  velocity, joint positions, joint velocities, previous actions and gravity direction.
- **Output:** 29 joint position targets for PD control at 50 Hz.

The intuition: the token says *what the next second of motion should look like*, and the decoder works
out *how to make this body do that from where it is now*. So a token is closer to an **intent** than to
a pose. Holding one fixed token means "keep doing this", not "freeze in this pose". That is why every
metric below separates **static** (stands still), **moving** (stays up but keeps walking or drifting)
and **fell**.

Language never reaches the decoder directly:

- Text → motion in the paper goes through **GEM**, which is external and not in this repo.
- The deploy stack's high-level interface is the **kinematic planner**: 27 `LocomotionMode`s plus
  movement direction, facing, speed and pelvis height. The planner generates a motion, the encoder
  turns it into tokens, and the decoder turns those into joint targets.
- Part 1 uses this planner vocabulary as "SONIC's built-in commands".

## Layout

| file | role |
|---|---|
| `common.py` | constants mirrored from the C++ deploy: joint orders, gains, action scale, default pose, FSQ grid |
| `make_scenes.py` | generates obstacle scenes `scene_inv_*.xml` next to `scene_43dof.xml` (already generated) |
| `metrics.py` | `EvalRecorder` logs root pose, contacts (floor / obstacle / self), torque saturation and falls at 50 Hz, then summarises |
| `probe_sim.py` | pure-Python closed loop: MuJoCo + `model_decoder.onnx` on CPU, same observation and PD as the C++ stack |
| `encoder_utils.py` | Python version of the robot-mode encoder input (used for real tokens and the tracking positive control) |
| `build_token_library.py` | builds `token_library.npz`, a set of real tokens (reference motions, plus C++ planner logs if added) |
| `token_sets.py` | hand-built token schedules: random, dim sweep, perturb, interp, held / replayed real tokens |
| `decoder_probe.py` | **batch part-2 measurement on CPU**: thousands of trials, parallel workers |
| `scenarios.yaml`, `run_planner_scenarios.py` | **part 1**: timed planner-command scripts and their ZMQ driver |
| `stream_tokens.py` | **part 2, visual**: streams token sets into the C++ deploy (ZMQ protocol v4, bypasses the encoder) |
| `run_part1.sh`, `run_part2.sh` | one-command tmux launchers: sim (viewer) \| C++ deploy \| driver |
| `parity_check.py` | checks the Python probe against a logged C++ run |
| `analyze.py` | reports and figures for probe, part1, part2 and library |

Changes to existing files are minimal and opt-in:

- `run_sim_loop.py` / `SimLoopConfig`: new `--robot-scene`, `--eval-log-dir` and `--no-fall-reset` flags.
- `base_sim.py`: an `EvalRecorder` hook that is only active with `--eval-log-dir`.

## Important: this is *not* `decoupled_wbc/docker/entrypoint/deploy.sh`

That script runs the older decoupled stand/walk WBC. SONIC sim2sim is two processes talking over DDS
on `lo`:

- `gear_sonic/scripts/run_sim_loop.py`: the MuJoCo sim at 200 Hz;
- `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`: TensorRT encoder and decoder, policy at 50 Hz,
  planner at 10 Hz.

`deploy.sh sim` runs the binary after an interactive prompt. Our launchers call the binary directly,
with the same arguments plus `--input-type zmq_manager` and logging flags.

## Setup (done once already)

```bash
cd gear_sonic/investigation/decoder_expressivity
PY=../../../.venv_sim/bin/python          # .venv_sim now also has onnxruntime + matplotlib
$PY make_scenes.py --list                 # scenes exist already; re-run to regenerate
$PY build_token_library.py                # already built (token_library.npz, ~1 min on CPU)
```

## Part 1: planner commands (needs the GPU for the C++ deploy)

```bash
./run_part1.sh --list                                   # scenario catalogue
./run_part1.sh step_slowwalk_02                         # hurdle_10cm by default
./run_part1.sh step_slowwalk_02 --scene hurdle_20cm     # same command, other obstacle size
./run_part1.sh gap_sideways_ledge                       # shimmy through a 32 cm opening
./run_part1.sh duck_crawl_h06                           # squat + crawl under an 85 cm bar
./run_part1.sh clutter_slow_zigzag
./run_part1.sh vocab_tour --scene flat                  # every planner mode; good for the token library
```

**What happens on screen:**

1. tmux opens three panes: sim (MuJoCo viewer), deploy, and driver (bottom-right).
2. Wait until the deploy pane is idle, then press **Enter** in the driver pane. The robot stands up while
   the elastic band holds it.
3. Press **9** in the MuJoCo viewer to drop the band, let the robot settle, then press **Enter** in the
   driver. The scenario plays.
4. **Ctrl-C** the driver (this sends `stop`), then **Ctrl-C** the sim (this flushes `sim/summary.json`).
   Close everything with `tmux kill-session -t sonic_eval`.

**Score the run:**

```bash
$PY analyze.py part1 runs/part1/<run>        # or several runs at once -> runs/analysis/part1/report.md
```

**Per-run metrics:**

- passed obstacle: the pelvis got 0.4 m past the obstacle plane without falling;
- time to cross;
- obstacle contact ticks, including trunk-on-obstacle (bar or wall hits);
- knee-on-floor contact, maximum tilt, falls;
- a figure with x, z and tilt over time, the planner step boundaries, and a top-down path with footprints.

**Things to watch visually:**

- **Hurdle:** does foot clearance ever exceed the ~5 cm normal swing? Toe stubs show up as obstacle contacts.
- **Gap:** does side-stepping keep the shoulders inside the opening?
- **Low bar:** head or torso contact with the bar.
- **Clutter:** does the gait adapt at all? The planner has no terrain input, so expect it to step blindly.

**Expected limits of the interface (known before running):**

- There is no knob for foot-lift height or foot placement.
- `height` is only honoured by the squat, kneel and crawl modes. Walking modes ignore it, so the only
  ways to walk lower are the stealth-walk styles or crawling.
- The planner never sees the scene.

## Part 2: hand-built tokens

### (a) Batch measurement on CPU (the main quantitative result; runs any time)

About 10× real time per core. With 16 workers, the full 2048-trial `dim_sweep` takes about 3 minutes.

```bash
$PY decoder_probe.py --set sanity --workers 1
$PY decoder_probe.py --set random_hold     --n 500 --workers 16    # whole cube: uniform random tokens
$PY decoder_probe.py --set random_marginal --n 500 --workers 16    # per-dim realistic, jointly random
$PY decoder_probe.py --set random_resample --n 200 --workers 16    # new random token every 1 s
$PY decoder_probe.py --set dim_sweep --workers 16                  # 64 dims x 32 levels around standing
$PY decoder_probe.py --set dim_sweep --bases ref_closed:walking_quip_360_R_002__A428 --workers 16
$PY decoder_probe.py --set perturb --n 25 --workers 16             # distance-from-data curve
$PY decoder_probe.py --set interp --n 100 --workers 16
$PY decoder_probe.py --set in_dist_hold --n 300 --workers 16       # freeze real tokens
$PY decoder_probe.py --set in_dist_replay --workers 8              # positive control
$PY analyze.py probe runs/probe/*                                  # -> runs/analysis/probe/report.md
$PY analyze.py library                                             # level usage of real tokens
```

**Trial protocol:**

1. Robot reset standing on the floor.
2. Standing token for 2 s.
3. Probe schedule (default: one token held for 6 s).
4. Standing token for 2 s, which measures whether the robot *recovers*.

**What gets measured:** survival time, outcome, pelvis height and tilt, displacement, settle speed and
joint-velocity RMS, contacts, torque saturation and recovery.

**Watching trials:** add `--viewer` to step through trials in a MuJoCo window at real time, for example:

- `--set dim_sweep --dims 7 --viewer`
- `--set random_hold --n 5 --viewer`

### (b) Visual check through the real C++ stack (needs the GPU)

```bash
./run_part2.sh --set dim_sweep --dims 7 --hold-s 3        # one dim through all 32 levels (~2.7 min)
./run_part2.sh --set random_hold --n 10 --hold-s 5
./run_part2.sh --set in_dist_replay --max-trials 3        # positive control: should dance/walk/squat
./run_part2.sh --set sanity --policy-input-log            # also records data for parity_check.py
$PY analyze.py part2 runs/part2/<run>                     # scores each trial from the sim eval log
$PY parity_check.py runs/part2/<run>                      # run once: is the CPU probe faithful?
```

The flow is the same as part 1. The driver starts control in planner/IDLE, you release the band, and
then the driver switches to streamed-motion mode and plays the tokens. There are 2 s of standing token
between trials.

## Validation done so far (CPU, without the C++ stack)

- **Standing token** (`LATENT_INITIAL_MOTION_TOKEN`) held for 10 s: the robot stands still, with pelvis
  at 0.79 m and joint-velocity RMS of 0.001.
- **Closed-loop tracking positive control:** Python encoder → tokens → decoder tracks all six shipped
  reference motions (squat, walk-and-turn, dance, kick, one-leg hop, macarena) with no falls and about
  0.09 rad mean joint error. The walking clip covers 5.1 m against the reference's 5.0 m.
- This checks the observation layout, joint orderings, gains and action scaling end to end.
- **Still to do:** `parity_check.py` against a real C++ log. It needs one GPU run, for example
  `run_part2.sh --set sanity --policy-input-log`.

## Preliminary numbers (mini smoke run, small n; re-run at full size)

| set | n | static | moving | fell |
|---|---|---|---|---|
| dim_sweep (6 dims × 32 levels, base = standing) | 192 | 69% | 28% | 4% |
| perturb (base standing) | 64 | 62% | 27% | 11% |
| interp between real tokens | 30 | 63% | 17% | 20% |
| in_dist_hold (real tokens frozen) | 32 | 47% | 38% | 16% |
| random_marginal (realistic per-dim values, random combination) | 32 | 6% | 41% | 53% |
| random_hold (uniform over the full 32⁶⁴ cube) | 32 | 0% | 16% | 84% |

Token-library statistics (real motion only):

- Each dim uses 15–27 of its 32 levels.
- The per-dim standard deviation is about 3 levels; uniform random would be 9.2.
- 98% of values lie in roughly [-9, 7].

**Early reading (to be confirmed at full size):**

- The decoder tolerates large edits to *one* dim, and small edits to many.
- Uniformly random tokens are far off the data and mostly fall within about 1.5 s.
- Stability drops sharply with L1 distance from real tokens; see `distance_to_library.png`.

## Caveats

- **The probe is not the C++ loop.** The CPU probe has no DDS or thread latency and holds the hands with
  a weak PD. The C++ sim run plus `parity_check.py` is the ground truth for whether the Python
  reproduction is faithful.
- **"Stable" depends on time.** A held walking token is *supposed* to keep walking. We report
  static / moving / fell rather than one bit.
- **Part 1 needs the GPU.** The planner runs only inside the C++ binary (TensorRT, 774 MB ONNX), so part 1
  cannot run on the CPU probe.
