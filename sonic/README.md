# GEAR-SONIC: a humanoid whole-body controller, timed in closed loop

GR00T (the rest of this repo) decides *what* a robot should do a few times a second. Something still has to keep the
robot balanced while it does it. **GEAR-SONIC** is NVIDIA's whole-body controller for the **Unitree G1** humanoid, from
[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (code Apache-2.0, weights on Hugging Face
as [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) under the NVIDIA Open Model License). Its C++ deploy program
`g1_deploy_onnx_ref` runs three networks:

- an **encoder** and a **decoder**, every control tick at **50 Hz** (one tick every 20 ms): robot state + reference motion in,
  29 joint targets out;
- a **planner**, at up to 10 Hz while walking or running, which turns keyboard commands (walk, turn, run) into a reference motion.

We ran that program unmodified, apart from a timing probe, against upstream's MuJoCo simulation of the G1, on an RTX 5090
desktop, a Jetson AGX Thor and a Jetson AGX Orin 64 GB. We asked whether each computer keeps the 20 ms rhythm, and whether
the robot stays up and follows its commands. The numbers are in [RESULTS.md §8](../RESULTS.md#8-gear-sonic-whole-body-controller-the-50-hz-loop-on-each-computer-measured-2026-10-13--2026-10-14)
and the README. Two follow-ups are below: a [controlled band-drop test](#controlled-band-drop-test) (RESULTS.md §8.1) and a
[late-start fix on the Jetsons](#late-start-fix-jetsons) (RESULTS.md §8.2).

## How it was measured

**One program, one span, one script.** Every machine ran upstream `g1_deploy_onnx_ref` at commit `b042411` (2026-09-23) with
the same probe ([harness/sonic_probe.hpp](harness/sonic_probe.hpp) + [harness/sonic_probe.patch](harness/sonic_probe.patch)).
The probe timestamps each control tick and copies tensors into preallocated memory: no I/O in the loop, written to disk at exit.
All three networks are fp32 TensorRT engines that upstream code builds on first launch on each machine (upstream's defaults;
TensorRT's TF32 permission is left on, see below). The same keyboard script drove every run (82 s):

| t (s) | 0 | 2 | 7 | 10 | 18–19 | 25 | 31 | 36 | 41 | 45 | 49 | 82 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| command | start (robot held by an elastic band) | **band off: robot dropped** | planner mode | walk | 3 × 30° turn | run | stop | slow walk | stop | reference mode | 27.5 s dance clip | exit |

**What is timed** (from the per-tick log, never from a self-reported counter):

- **step** = from the start of observation building to after the frame advance: robot-state gather, observation build,
  encoder, decoder, hand commands, ZMQ publish. A **compute miss** is a step over 20 ms.
- **gap** = start-to-start time between consecutive ticks. A **late start** is a gap over 21 ms; a **slot miss** is a gap over 25 ms.
- **planner** = upstream's own TensorRT model time per planner call (`last_timing_.model_duration`), plus the whole UpdatePlanning wall.

**What is judged** (from the simulator's ground truth):

- **fall, rule window** (pre-registered): pelvis below 0.30 m, tilt above 60°, or a simulator fall-reset, from 1 s after
  the band drop to the end of the run.
- **band-drop fall** (found during the Jetson runs, reported separately): a simulator fall-reset within 3 s of the band-off
  command. A reset later than +1.0 s is counted in both columns.
- **fall, D1 rule** (written before the controlled band-drop test; used from then on, and the first campaign re-scored with it):
  any simulator fall-reset from the band-off command to the stop command, split into drop window (≤ 3 s after the command) and later.
- **tracking**: mean joint error per phase within ±30% of the RTX 5090 runs' range, distance walked per phase within ±15%,
  turn angle within ±15°.

**Runs.** RTX 5090: 3 runs (v1_r1–v1_r3), the reference. Each Jetson: 10 unloaded runs (cl_r1–r3 pre-registered, cl_r4–r10
added after one Thor run fell at the band drop, disclosed), then two loaded arms of 3 runs each: **CPU saturated**
(`stress-ng --cpu <ncores> --cpu-method matrixprod`) and **GPU co-tenant** (GR00T N1.5 on NVIDIA's full TensorRT path calling
`get_action` back-to-back on the same GPU, [harness/groot_cotenant.py](harness/groot_cotenant.py)). One diagnostic run per Jetson
(n = 1) repeated the CPU-saturated arm with the deploy's SCHED_FIFO request disabled.

**Engine correctness (A2).** Before trusting a Jetson's closed loop, its upstream-built engines were run on the exact tensors
the RTX 5090 consumed: 4,099 encoder/decoder ticks from run v1_r1 and 86 planner calls ([gate/](gate/)). Encoder and decoder
match. The planner misses the pre-registered gate on both Jetsons (per-call max error p90 0.109 rad on Thor, 0.075 rad on Orin,
against 0.05 rad). A `--noTF32` build of the same ONNX on the same board matches fp32 to under 1e-4 rad, so the cause is
TF32, which upstream leaves on, not the TensorRT version. The Orin has TensorRT 10.3 where upstream asks for 10.7. No
closed-loop consequence was seen.

**Pinned or not.** Jetsons: nvpmodel MAXN + `jetson_clocks`, read back. RTX 5090: default boost, not locked.

## Same-span, not same-everything

The rows compare the same program, probe, span, script and scoring. Four things still differ, and each is disclosed in the records:

- **Root vs non-root.** On the Jetsons the deploy ran as root, so upstream's main thread gets SCHED_FIFO 99 on CPU0 and
  busy-spins there (upstream calls `sleep(0.02)` on an integer `sleep()`, which is `sleep(0)`). On the RTX 5090 it ran
  non-root in a container, where that FIFO request fails silently.
- **Simulator command intake.** On the Jetsons the simulator polled its DDS command readers once per physics step
  (`SIM_POLL_CMDS=1` in [harness/sim_driver.py](harness/sim_driver.py)). The RTX 5090 used upstream's per-sample Python listener.
- **Clocks**, as above.
- **The simulator shares the computer.** On a robot the physics is not on the computer. That makes the CPU-saturated arm
  pessimistic for the simulator side: the hogs starve it too, and the per-run simulator overrun steps are recorded.

## Controlled band-drop test

Some unloaded first-campaign runs fell right after the band drop (Thor 3/10, Orin 1/10, RTX 5090 0/3) with clean timing. The
simulator read its stdin every 100 ms, so the release came 0–100 ms after the command, uncontrolled. The follow-up controls it:

- **Release mechanism.** `sim_driver.py` with `SIM_RELEASE_DELAY_MS=D` polls stdin every 5 ms physics step; the orchestrator
  (`--release_delay_ms D`) writes `band_off <its CLOCK_MONOTONIC ns>`, and the simulator releases the band at the first physics
  step at or after command + D. It logs the actual delay, the pelvis tilt and the horizontal pelvis speed at that step
  (`release` in each `__run.json`). A simulator-only self-test at D = 10 and 25 ms ran before the runs on each machine.
- **Design.** D ∈ {10, 25, 40, 60, 80, 100} ms × 3 runs per machine, unloaded, NVIDIA's stock deploy (same binaries as the first
  campaign), in a seeded shuffled order per machine ([harness/banddrop_order.json](harness/banddrop_order.json)), 15 s apart.
  Predictions and decision rules were written before the first run. Run names: `bd_D<ms>_r<n>`.
- **Scoring.** Falls by the D1 rule. Two checks added afterwards and labelled as such: a body-state check (tilt above 60° or pelvis
  below 0.30 m after the command), because the simulator resets only below 0.2 m and a fallen robot can lie just above that for
  seconds; and simulator-state fingerprints (`sim_fingerprint` in each `__run.json`) to find runs that repeat bit for bit.
- **Statistics** (DERIVED, in `make_sonic_table.py`, numpy-only): Clopper-Pearson 95% intervals, Fisher exact two-sided tests,
  an exact permutation test of the rank sum (exploratory).

Result: every machine falls at some delay (3/18 each), with on-time control and no shown dependence on D. See RESULTS.md §8.1.

## Late-start fix (Jetsons)

On the Jetsons the deploy runs as root, so upstream's main thread gets SCHED_FIFO 99 pinned to CPU0, and its `sleep(0.02)` on an
integer `sleep()` makes it spin, while the 50 Hz control thread runs at normal priority. The fix is a deployment change:
**control thread SCHED_FIFO 90 + sleeping main loop instead of upstream's busy-spinning FIFO-99 main thread**
([harness/latefix.patch](harness/latefix.patch), applied on top of `sonic_probe.patch`). It is inert unless the deploy gets
`SONIC_CTRL_FIFO=90` and `SONIC_MAIN_SLEEP=abs`, and it needs root or CAP_SYS_NICE. The first campaign's three arms were repeated
with it (3 runs each per Jetson, `fix_cl_r*`, `fix_a4cpu_r*`, `fix_a4gpu_r*`), after one same-session stock run (`stockref_cl_r1`).
The orchestrator reads every deploy and simulator thread's scheduling policy, priority and CPU time from /proc after init, at
t + 20 s and at the stop command (`levers_readback.json`), which is how we checked that the fix was in effect in every run.
Result: RESULTS.md §8.2.

## Files

| Path | What |
|---|---|
| `harness/sonic_probe.hpp`, `harness/sonic_probe.patch` | the timing probe (header + patch to upstream `g1_deploy_onnx_ref.cpp` @ b042411). The 5090 runs used the same probe without its planner-I/O capture; per-tick record and span are identical |
| `harness/sim_driver.py` | headless upstream MuJoCo G1 simulator: stdin commands (band_off), 50 Hz state log, fall-reset counter, wall-clock pacing, optional `SIM_POLL_CMDS=1`, optional controlled release `SIM_RELEASE_DELAY_MS=D` |
| `harness/run_closed_loop_x86.py`, `harness/Dockerfile` | one run on an x86 + NVIDIA GPU host, deploy in a container with TensorRT 10.13 (upstream's x86 requirement) |
| `harness/run_closed_loop_jetson.py`, `harness/build_jetson.sh`, `harness/mkref.py`, `harness/jetson_runs.sh` | native Jetson build, one run, and the campaign as we ran it (unloaded, CPU-saturated, GPU co-tenant, diagnostic, `banddrop`, `fix`) |
| `harness/banddrop_order.json`, `harness/banddrop_x86.sh` | controlled band-drop test: the seeded run order per machine, and the RTX 5090 batch |
| `harness/latefix.patch` | the late-start fix (control thread SCHED_FIFO 90 + sleeping main loop), a patch to upstream `g1_deploy_onnx_ref.cpp` on top of the probe patch, plus a small header |
| `harness/groot_cotenant*.{py,sh}` | the GR00T N1.5 full-TensorRT co-tenant loop (uses this repo's GR00T setup and `speed/` engines) |
| `scoring/analyze_run.py`, `scoring/analyze_cl.py` | one run dir → `summary.json` (latency, phases, falls, heading, distance) |
| `scoring/export_run.py` | one analysed run dir → the published raw format in `results/sonic/raw/` (with the release readback, body-state check and state fingerprints) |
| `scoring/score_groot.py` | GR00T co-tenant rate alone vs during each SONIC run |
| `gate/` | engine-correctness gate: reference sets (`make_refsets.py`), board engines on them (`engine_gate.py`, `planner_gate.py`), scoring (`score_gate.py`), a harness check (`planner_ort_check.py`) |
| `make_sonic_table.py` | rebuilds `results/sonic/sonic_same_span_table_2026-10-14.json` from `results/sonic/raw/`, including the controlled band-drop analysis and the late-start-fix groups |
| `env.sh` | paths (`SONIC_WORK`, `SONIC_REPO`, ...) |

Results: `results/sonic/sonic_rtx5090_2026-10-13.json`, `sonic_thor_2026-10-14.json`, `sonic_orin_2026-10-14.json` (setup,
versions, engine hashes, A2, loaded arms, predictions vs outcomes, caveats); `sonic_banddrop_controlled_2026-10-14.json` and
`sonic_jetson_latefix_2026-10-14.json` (the two follow-ups); `sonic_same_span_table_2026-10-14.json`
(every run and every group, rebuilt); `raw/<board>/<run>__ticks.csv.gz` (per tick: time, step, encoder, decoder, planner
active), `__planner.csv.gz` (per planner call), `__run.json` (band-off and stop times, simulator resets, fall flags,
tracking per phase; for the follow-up runs also the release readback, body state and state fingerprints). All times are
relative to the control start. Run names: `v1_r*` / `cl_r*` unloaded, `a4cpu_r*` CPU saturated, `a4gpu_r*` GPU co-tenant,
`diag_nonroot_cpu`, `bd_D<ms>_r*` controlled band drop, `stockref_cl_r1` and `fix_*` the late-start fix.

**Not published:** the per-tick input/output tensors (about 50 MB per run), the simulator state logs, the two reference
sets (their sha256 is in the RTX 5090 record), deploy logs, host census files and the per-run /proc scheduling readbacks
(summarised per run in the late-start-fix record). The joint-error and fall numbers derived
from them are published per run, as measured.

## Rebuild the table (no GPU)

```bash
python3 sonic/make_sonic_table.py /tmp/sonic_table.json   # every count and percentile, from the per-tick logs
python3 tools/check_results.py                             # also compares it with the committed table, RESULTS.md and README
```

## Run it yourself

What we did, in order. Paths come from `sonic/env.sh` (`source setup/env.sh; source sonic/env.sh`).

1. **Upstream.** `git clone https://github.com/NVlabs/GR00T-WholeBodyControl $SONIC_REPO && git -C $SONIC_REPO checkout $SONIC_COMMIT`,
   then **`git lfs pull`**. Without LFS, `gear_sonic_deploy/reference/example/*.csv` are 3-line pointer files. The C++
   loader accepts them silently as 2-timestep motions, and control then fails at the first tick.
2. **Models.** `python download_from_hf.py` in the checkout (upstream's script; `nvidia/GEAR-SONIC`). We used
   `policy/release/model_{encoder,decoder}.onnx` and `planner/target_vel/V2/planner_sonic.onnx`; their sha256 are in every record.
3. **Probe.** `cp sonic/harness/sonic_probe.hpp $SONIC_DEPLOY/src/g1/g1_deploy_onnx_ref/src/ && patch -d $SONIC_REPO -p1 < sonic/harness/sonic_probe.patch`.
   The probe records only when `SONIC_PROBE_DIR` is set.
4. **Build.** x86: `docker build -t sonic-deploy:trt1013 sonic/harness`, extract the TensorRT 10.13.0.35 tarball to
   `$SONIC_TRT`, and build with upstream's `just`/CMake inside the container. If the container runs with `--user`, mount
   `/etc/passwd`, because the Unitree SDK calls `getpwuid`. Jetson: `bash sonic/harness/build_jetson.sh` (board TensorRT,
   onnxruntime 1.16.3 aarch64 in `$SONIC_ORT`).
5. **Simulator Python.** mujoco 3.15, upstream `gear_sonic` importable (`PYTHONPATH=$SONIC_REPO`), `unitree_sdk2py` and
   `cyclonedds` 0.10.2 (set `CYCLONEDDS_HOME` if you built it yourself).
6. **Run.** x86: `python3 sonic/harness/run_closed_loop_x86.py --run_dir $SONIC_RUNS/v1_r1 --script v1`. Jetson, as root:
   `sudo -E bash sonic/harness/jetson_runs.sh unloaded 1 3` (then `cpu`, `gpu <orin|thor>`, `diag`). The first launch builds
   the engines (minutes); let it finish before measuring. Stop gracefully only: the scripts send `o`, then SIGINT, then SIGTERM.
   Never SIGKILL a process holding a CUDA context.
7. **Controlled band drop.** x86: `bash sonic/harness/banddrop_x86.sh`. Jetson, as root: `sudo -E bash sonic/harness/jetson_runs.sh banddrop <orin|thor>`.
8. **Late-start fix (Jetson).** Keep the stock binary, then `patch -d $SONIC_REPO -p1 < sonic/harness/latefix.patch`, rebuild with
   `build_jetson.sh`, copy `target/release/g1_deploy_onnx_ref` to `$SONIC_FIX_BIN`, and restore the stock source and binary
   (`patch -R` and rebuild). Then, as root: `sudo -E SONIC_FIX_BIN=... bash sonic/harness/jetson_runs.sh fix <orin|thor>`.
9. **Score.** `python3 sonic/scoring/analyze_cl.py <run_dir>`, then
   `python3 sonic/scoring/export_run.py <run_dir> results/sonic/raw/<label> <label>`, then `python3 sonic/make_sonic_table.py`.
   Tracking is scored against the RTX 5090 runs already in `results/sonic/raw/rtx5090/`.

Two traps we hit, both handled by the scripts:

- **Simulator pacing.** Upstream's rate limiter (`sleep(dt - elapsed)`) runs slower than real time (about 0.82× on our host),
  while the controller runs on the wall clock. `sim_driver.py` paces to absolute deadlines.
- **Reference-motion order.** The deploy loads `reference/example/` in directory-iteration order. The first motion loaded
  sets the stand pose, and the script picks the dance clip by index. `mkref.py` recreates the directory so the order
  matches our RTX 5090 runs.

## Credit and license

GEAR-SONIC and `g1_deploy_onnx_ref` are NVIDIA's work: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
source under the Apache License 2.0, model weights under the NVIDIA Open Model License. Nothing of theirs is vendored
here. `harness/sonic_probe.patch` and `harness/latefix.patch` are modifications of their `g1_deploy_onnx_ref.cpp` and are
distributed under the Apache License 2.0 (see [NOTICE](../NOTICE)). `harness/sim_driver.py` drives their `gear_sonic` simulator through its
public API. The MuJoCo G1 model, the motion clips and the network weights are downloaded from upstream, not redistributed.
