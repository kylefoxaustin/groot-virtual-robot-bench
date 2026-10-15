# GEAR-SONIC on the Qualcomm IQ-9075 (Hexagon v73 NPU)

The scripts to run NVIDIA's GEAR-SONIC whole-body controller ([sonic/README.md](../../sonic/README.md)) on your own IQ-9075:
the encoder and decoder on one Hexagon NSP every 20 ms, the planner split between the second NSP and the CPU, upstream's
C++ deploy program cross-built with a QNN shim in place of TensorRT, and the same MuJoCo closed loop, keyboard script,
controlled band-drop test and scoring that the RTX 5090, Jetson AGX Thor and Jetson AGX Orin runs used.
**No IQ-9075 results are published in this repo** -- run it and see.

Like the GR00T scripts in [iq9/](../README.md), these are the research scripts, cleaned up: private paths became
environment variables, and the board-side pieces were rewritten to share the Jetson harness. Read each script's header
before running it. Qualcomm's SDKs are not included and never redistributed: the C++ code only `#include`s QNN headers
and compiles the SDK's SampleApp helper sources from **your** SDK install ([NOTICE](../../NOTICE)).

## What you need

| Thing | Version we used | Where it comes from |
|---|---|---|
| Qualcomm IQ-9075 EVK (Dragonwing, two Hexagon v73 NSPs), Yocto image with fastrpc, root ssh with key auth | -- | Qualcomm / Thundercomm |
| QAIRT / QNN SDK: x86 converter + context-binary generator, board runtime (`libQnnHtp.so`, `libQnnSystem.so`, v73 skel) | **2.47.0.260601** | [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) or the [Qualcomm Software Center](https://softwarecenter.qualcomm.com) (Qualcomm account + license acceptance; not redistributed here) |
| A Python tree on the board (numpy; onnxruntime for the CPU control arm) | Python 3.12 | your choice; the simulator's packages are installed next to it by `deploy/stage_board.sh` |
| An x86 Ubuntu host: `aarch64-linux-gnu-gcc-11`/`g++-11`, cmake, curl, rsync, Eigen 3 headers, `libmsgpack-dev`, `nlohmann-json3-dev`, `libzmq3-dev`; python3 with onnx, onnxruntime, numpy | Ubuntu 22.04 | your distro |
| onnxruntime aarch64 release (the deploy and `sonic_rt` link it; it is copied to the board) | **1.20.1** | [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) (`onnxruntime-linux-aarch64-1.20.1.tgz`) |
| Upstream GEAR-SONIC checkout and models, as for the NVIDIA machines | commit `b042411` | [sonic/README.md, "Run it yourself"](../../sonic/README.md#run-it-yourself), steps 1-2 (including `git lfs pull`) |
| A reference closed-loop run with the probe (for the reference sets) | -- | an NVIDIA machine per [sonic/README.md](../../sonic/README.md), or the IQ-9075's own CPU control arm (step 8) |

**Paths** come from three env files, in this order:

```bash
source setup/env.sh            # QAIRT_SDK_ROOT (x86 SDK root, contains 2.47.0.260601/), IQ9_QNN_ROOT, IQ9_PY_ROOT (board paths)
source sonic/env.sh            # SONIC_WORK, SONIC_REPO, SONIC_DEPLOY, ...
export IQ9_HOST=<board>        # ssh target of the IQ-9075
source iq9/sonic/env.sh        # IQ9_SONIC_WORK (host work dir), IQ9_SONIC_BOARD_DIR (board dir, default /opt/sonic),
                               # IQ9_SONIC_ORT, IQ9_SONIC_SYSROOT, IQ9_SONIC_SAMPLEAPP, IQ9_PYVER, IQ9_CXX
```

On the board, `deploy/stage_board.sh` writes `$IQ9_SONIC_BOARD_DIR/env.sh`; source it in every board shell.
**Never SIGKILL a process that holds an NSP** (`g1_deploy_onnx_ref_qnn`, `sonic_rt`, `sonic_bench`, `qnn-net-run`): a killed
DSP session can leave the NPU degraded (slower, plausible-looking numbers) until reboot. Every script here stops the
board side with stdin `o`, SIGINT or SIGTERM.

## How the port works

| Network | Where it runs | How |
|---|---|---|
| encoder (FSQ tokens) | NSP0, fp16 | `sonic_enc_core`: the upstream encoder cut just **before** its FSQ `Round`; the CPU does `round(x)/16` and picks the active sub-encoder (`encdec/make_graphs.py`, `sonic_host.py`, `deploy/shim/encoder.hpp`) |
| decoder (29 joint targets) | NSP0, fp16 | the upstream decoder unchanged (`sonic_dec`) |
| planner, transformer and conv bulk | NSP1, fp16 | four blocks S, R, P, D cut out of `planner_sonic.onnx` (`planner/build_blocks.py`), dilated Conv1d rewritten (`fix_dilated.py`) |
| planner, ArgMax / TopK / OneHot / CumSum / ScatterND / seed hash | CPU, onnxruntime fp32 | five host stages H0..H4 between the blocks (`planner/build_host.py`) |

All context binaries are built for v73 with `fp16_relaxed_precision`, `advanced_activation_fusion: false`, 8 MB VTCM and
O3. In the deploy, `deploy/shim/` replaces upstream's TensorRT encoder, decoder and planner classes with QNN ones (same
class names and public API), so `g1_deploy_onnx_ref.cpp` keeps only this repo's probe and late-start patches. The NSP for
each network is chosen with `SONIC_QNN_{ENC,DEC,PLANNER}_DEV` (strict: an unknown device id is an error, never a fallback).

## Steps

### 1. Upstream, models and reference sets

Follow [sonic/README.md](../../sonic/README.md#run-it-yourself) steps 1-2 on the x86 host (clone at `$SONIC_COMMIT`,
**`git lfs pull`**, `download_from_hf.py`). Then make the two reference sets from a closed-loop run made with the probe:

```bash
python3 sonic/gate/make_refsets.py encdec  <reference run dir> $IQ9_SONIC_WORK/refset_encdec.npz
python3 sonic/gate/make_refsets.py planner <reference run dir> $SONIC_DEPLOY/planner/target_vel/V2/planner_sonic.onnx $IQ9_SONIC_WORK/refset_planner.npz
```

The encdec set is every encoder/decoder input and output of that run; the planner set is every planner call's inputs plus
an onnxruntime fp32 run of the full planner on them, which is what the planner gate scores against.

No NVIDIA machine? Build and stage everything first (steps 2-4 without their board tests), then run the CPU control arm
(`iq9_runs.sh cpuarm`, step 6): its probe output is a run of the upstream encoder/decoder in fp32 on the board's CPU, and its
planner I/O log feeds `make_refsets.py planner`, whose gate reference is onnxruntime fp32 either way.

### 2. Encoder and decoder: export, compile, test on the board

```bash
bash iq9/sonic/encdec/export.sh                                   # make_graphs.py, check_core_equiv.py, build_ctx.sh (aaf0)
bash iq9/sonic/deploy/get_sampleapp.sh                            # SDK SampleApp sources -> $IQ9_SONIC_SAMPLEAPP (not in this repo)
bash iq9/sonic/encdec/build_bench.sh                              # sonic_bench, the resident enc/dec runner
```

`check_core_equiv.py` must report the split as exact: `sonic_enc_core` plus the CPU `Round` reproduces the full upstream
encoder bit for bit in fp32. After the board is staged (step 4), `encdec/run_encdec.sh $IQ9_SONIC_WORK/refset_encdec.npz`
writes the fp16 prediction on the host first (`fp16_predict.py`: fp32, fp16-storage and fp16-accumulate brackets on your
exact inputs), then runs on the board: correctness on NSP0, a planted-input run (must fail the gate), a wrong-observation
run, the CPU onnxruntime fp32 arm, and latency back-to-back and paced at 50 Hz (with and without the NPU performance vote).
`gate.py` scores each against thresholds derived from the prediction; `loop_check.py` checks the actions produced inside the
timed loop. **Expected:** the NPU tokens and actions land inside the predicted fp16 bracket, the planted run fails, and the
encoder plus decoder fit comfortably inside the 20 ms tick.

### 3. Planner: split, compile, test on the board

```bash
bash iq9/sonic/planner/split.sh $IQ9_SONIC_WORK/refset_planner.npz   # blocks + host stages + rewrites + CPU check + context bins + manifest/calls
cd $IQ9_SONIC_WORK/planner && python3 $GVRB_ROOT/iq9/sonic/planner/fp16sim.py        # fp16-emulating copies of the blocks (blocks_sim/)
python3 $GVRB_ROOT/iq9/sonic/planner/predict.py $IQ9_SONIC_WORK/refset_planner.npz pred.json
python3 $GVRB_ROOT/iq9/sonic/planner/add_thresholds.py pred.json        # the gate, written BEFORE any board run
```

`test_chain.py` (run by `split.sh`) must show the host stages plus fp32 blocks equal to the full planner. After staging
(step 4): `bash iq9/sonic/planner/run_planner_gate.sh $IQ9_SONIC_WORK/refset_planner.npz $IQ9_SONIC_WORK/planner/pred.json`
runs every logged call on NSP1 with the blocks dumped, and again with a planted fault in block P (must fail). On the host,
`gate.py` scores it, `blkcheck.py` compares each block's board output with onnxruntime fp32 on the board's own block inputs
(this is how a block the NPU computes wrongly shows up), and `integrity.py` re-runs the host stages on the host from the
board's block outputs. **Expected:** a planner call finishes well inside its 100 ms slot (the planner runs at up to 10 Hz).

To hold the IQ-9075 to exactly the gate the Jetsons' TensorRT engines were held to (same reference sets, same thresholds),
convert its outputs and use the public scorer:

```bash
python3 iq9/sonic/gate/to_gate_npz.py $IQ9_SONIC_WORK/refset_encdec.npz $IQ9_SONIC_WORK/encdec_test/board/npu_aaf0 \
    $IQ9_SONIC_WORK/encdec_test/board/npu_wrongobs $IQ9_SONIC_WORK/planner/runs/correct_nsp1 iq9_gate.npz
python3 sonic/gate/score_gate.py iq9_gate.npz iq9_gate.json $IQ9_SONIC_WORK/refset_encdec.npz $IQ9_SONIC_WORK/refset_planner.npz
```

That gate was written for fp32 engines; an fp16 build is judged differently by the prediction-based gate above. Apply the
same gate to every machine you compare.

### 4. Cross-build and stage the board

```bash
bash iq9/sonic/deploy/build_sysroot.sh     # msgpack/json/cppzmq headers, libzmq 4.3.4 cross-built, the board's libz
bash iq9/sonic/deploy/build_clayer.sh      # cyclonedds 0.10.2 C extension for the board Python, against upstream's aarch64 libddsc
bash iq9/sonic/deploy/build_deploy.sh      # g1_deploy_onnx_ref_qnn: upstream + probe + latefix patches + QNN shim
bash iq9/sonic/rt/build_rt.sh              # sonic_rt: enc/dec loop + split planner in one process (loaded test, planner gate)
bash iq9/sonic/deploy/stage_board.sh       # board tree -> $IQ9_HOST:$IQ9_SONIC_BOARD_DIR, simulator wheels installed there
```

`build_deploy.sh` compiles a copy of upstream's `g1_deploy_onnx_ref` source with `sonic/harness/sonic_probe.patch` and
`sonic/harness/latefix.patch` applied and the shim headers dropped into its `include/` (upstream includes them by relative
path). The late-start code is inert unless its environment variables are set, so the one binary serves both the stock
and the late-start runs. The simulator on the board is the public `sonic/harness/sim_driver.py` and `mkref.py`.

### 5. Loaded test without the simulator (optional)

After step 2's board test (it stages the inputs in `$IQ9_SONIC_BOARD_DIR/data/encdec`), on the board as root:
`. $IQ9_SONIC_BOARD_DIR/env.sh && nohup sh $IQ9_SONIC_BOARD_DIR/rt/loaded_arms.sh <outdir> [fix] &`.
Five arms of 10,000 paced ticks each: baseline, planner at 10 Hz on the same NSP, planner on the other NSP, CPU saturated
(stress-ng, one hog per core), planner on the other NSP plus CPU saturated. Score with
`python3 iq9/sonic/rt/gaps.py <outdir>/{a,b,c,d,e}`: compute miss = step over 20 ms, late start = start-to-start gap over
21 ms, slot miss = gap over 25 ms, all from the per-tick log.

### 6. Closed loop on the board

On the board, as root (SCHED_FIFO needs root or CAP_SYS_NICE), detached:

```bash
. $IQ9_SONIC_BOARD_DIR/env.sh; H=$IQ9_SONIC_BOARD_DIR/harness
nohup bash $H/iq9_runs.sh smoke &          # stand, band drop, stop: check the whole chain first
nohup bash $H/iq9_runs.sh unloaded 1 3 &   # cl_r1..r3: the 82 s keyboard script, unloaded
nohup bash $H/iq9_runs.sh cpu &            # a4cpu_r1..r3: CPU saturated
nohup bash $H/iq9_runs.sh cpuarm &         # encoder+decoder on the CPU (onnxruntime fp32): the control arm
nohup bash $H/iq9_runs.sh diag &           # CPU saturated with the deploy's SCHED_FIFO request disabled
```

`harness/run_closed_loop_iq9.py` is the Jetson orchestrator with the board's paths: the simulator on the board's CPU with
`SIM_POLL_CMDS=1`, the deploy with encoder/decoder on NSP0 and planner on NSP1, the same keyboard script, the
`/proc` scheduling readback (`levers_readback.json`) and a load census before and after each run. Every run is refused on a
busy board (`harness/board_quiet.sh`) and starts 15 s after the previous one.

### 7. Late-start settings

Add `fix` to `unloaded`, `cpu` or `banddrop` (`iq9_runs.sh unloaded 1 3 fix`, `iq9_runs.sh cpu fix`). That sets
`SONIC_CTRL_FIFO=90` (the 50 Hz control thread raises itself to SCHED_FIFO 90), `SONIC_MAIN_SLEEP=abs` (upstream's main
thread sleeps to absolute 20 ms deadlines instead of busy-spinning on CPU0 at FIFO 99), `SONIC_QNN_PLANNER_THREADS=1`
(one onnxruntime thread for the planner's host stages) and the `performance` cpufreq governor for the batch (restored on
exit). Check in each run's `levers_readback.json` that the control thread really is SCHED_FIFO 90. **Expected:** without
these settings you may see occasional late ticks under planner load, because upstream gives SCHED_FIFO only to its idle
main thread and the control thread competes with the planner's CPU work at normal priority.

### 8. Controlled band-drop test

`nohup bash $H/iq9_runs.sh banddrop <seed> [fix] &`: release delays D = 10, 25, 40, 60, 80, 100 ms after the band-off
command, 3 runs each, unloaded, in a `random.Random(<seed>).shuffle` order (choose and write down the seed before the first
run). The simulator releases the band at the first physics step at or after command + D and logs the actual release
(`release` in the exported run). Method and scoring as in [sonic/README.md](../../sonic/README.md#controlled-band-drop-test).

### 9. Score

Copy the run directories to the host, then per run:

```bash
python3 sonic/scoring/analyze_cl.py <run_dir>                          # summary.json: latency, misses, falls, heading, distance
python3 sonic/scoring/export_run.py <run_dir> <out>/iq9 iq9             # the published raw format (ticks, planner calls, run.json)
```

`sonic/make_sonic_table.py` builds the three published machines from `results/sonic/raw/`; to put your IQ-9075 runs
beside them, add `"iq9"` to its `BOARDS` and `LABEL` in a local copy and point it at your export. Tracking is scored
against the RTX 5090 runs in `results/sonic/raw/rtx5090/`, within the method's tolerances (joint error per phase within
±30% of the RTX 5090 range, distance per phase within ±15%, turn within ±15°). Falls are counted by the same rules
(simulator fall-reset, and the body-state check for a robot lying just above the reset height).

## Method notes (traps we hit)

- **FSQ `Round` on the NPU.** The encoder ends in FSQ quantization (Tanh, scale, `Round`). In fp16 a value near a rounding
  boundary can round to the neighbouring level and flip a token, so the NPU graph stops before `Round` and the CPU rounds
  (round-half-even, then /16, then the sub-encoder select). `fp16_predict.py` reports how close your inputs come to a boundary.
- **Activation fusion.** Build every context binary with `advanced_activation_fusion: false`. With the default, the fp16
  SiLU MLPs came out wrong on v73. Also keep `graph_names` in the HTP config equal to the DLC basename, or the whole config
  block (VTCM size, precision, fusion) is ignored silently.
- **Dilated Conv1d.** The v73 HTP backend computed the planner's dilated 1-D convolutions (blocks R and D) wrongly in
  fp16, with no error and plausible-looking output; the planner gate failed and `blkcheck.py` localised it. `fix_dilated.py`
  rewrites each one exactly: where the dilation reaches past the sequence only the centre tap sees data, so it becomes a
  k=1 Conv with the centre weights; otherwise Pad by the dilation, three shifted Slices, Concat on channels, and one k=1 Conv
  with the three kernel taps stacked. A Resize rewrite (`fix_resize.py`) was tried first and is exact, but it was not the cause.
- **DDS listener lag.** Upstream's MuJoCo bridge receives `LowCmd` through a Python DDS listener callback per sample. On the
  board's CPU it fell behind the deploy's 500 Hz command stream, so the simulator applied commands late and the robot fell.
  `SIM_POLL_CMDS=1` in `sonic/harness/sim_driver.py` instead reads the latest sample of each command reader once per physics
  step; the orchestrator always sets it.
- **No aarch64 cyclonedds wheel.** `deploy/build_clayer.sh` builds the Python package's C extension against the libddsc that
  upstream ships for aarch64, so the simulator and the deploy use the same DDS library.
- **Reference-motion order.** The deploy loads `reference/example/` in `directory_iterator` order; the first motion sets the
  stand pose and the keyboard script picks the dance clip by index. `mkref.py` recreates the directory on tmpfs in the
  reference order and verifies it (the orchestrator refuses to run otherwise).
- **LFS pointer files.** Without `git lfs pull`, upstream's reference CSVs are 3-line pointer files that the C++ loader
  accepts silently as 2-timestep motions, and the aarch64 `libunitree_sdk2.a` / `libddsc.so` are pointer files that fail at
  link time. The build scripts check the libraries; check the CSVs yourself.
- **dmesg ring buffer.** Counting new `dmesg` lines to detect kernel messages during a run sees nothing once the ring buffer
  is full (the line count stays constant). `rt/rt_run.sh` selects messages by timestamp instead.
- **Root for SCHED_FIFO.** Upstream asks for SCHED_FIFO 99 on CPU0 for its main thread; as root it gets it and, because
  upstream's `sleep(0.02)` truncates to `sleep(0)`, that thread busy-spins its core. Without root the request fails silently
  and the late-start settings (`SONIC_CTRL_FIFO=90`) cannot take effect. Run the closed loop as root and read
  `levers_readback.json` to see what each thread actually got.
- **Measure lateness from the log.** `sonic_bench`'s `late_us` only counts a tick that was already late before its pacing
  sleep; a wake-up delay after the sleep is not in it. Use start-to-start gaps (`rt/gaps.py`, and `analyze_cl.py` for the
  deploy's per-tick probe log).
- **Quiet board.** A leftover process on an NSP or a busy CPU changes every number; every board script checks first, and a
  killed DSP session (SIGKILL) can leave an NSP slow until reboot.

## Files

| Path | What |
|---|---|
| `env.sh` | paths (host work dir, board dir, onnxruntime, sysroot, SampleApp copy, board Python version, cross compiler) |
| `encdec/make_graphs.py`, `check_core_equiv.py`, `sonic_host.py` | encoder cut before FSQ `Round` + decoder copy; CPU exactness check; host Round/select |
| `encdec/build_ctx.sh`, `export.sh` | ONNX -> DLC -> v73 fp16 context binary (aaf0); the export driver |
| `encdec/fp16_predict.py`, `gate.py`, `loop_check.py`, `lat_summary.py` | fp16 prediction on your inputs, the gate derived from it, loop-output check, latency summary |
| `encdec/sonic_bench.cpp`, `build_bench.sh`, `npz_to_f32.py`, `make_plant.py`, `cpu_ort_run.py`, `run_encdec.sh` | resident enc/dec runner (correct / latency), inputs, planted faults, CPU fp32 arm, the board test driver |
| `planner/spec.json`, `dump_vals.py`, `build_blocks.py`, `build_host.py`, `fix_resize.py`, `fix_dilated.py`, `test_chain.py`, `chain.py`, `inputs.py`, `make_manifest.py`, `build_ctx.sh`, `split.sh` | the planner split, its rewrites, the CPU exactness check, context bins, manifest and call file |
| `planner/fp16sim.py`, `predict.py`, `add_thresholds.py`, `gate.py`, `pmetrics.py`, `blkcheck.py`, `integrity.py`, `run_planner_gate.sh` | fp16 prediction, gate, per-block and pipeline checks, the board test driver |
| `rt/sonic_rt.cpp`, `rtctl.hpp`, `build_rt.sh`, `rt_run.sh`, `loaded_arms.sh`, `gaps.py` | enc/dec + split-planner runtime (planner gate, loaded test), env scheduling knobs, runners, gap scoring |
| `deploy/shim/` | QNN replacements of upstream's encoder, decoder (policy) and planner classes + the QNN runtime; `override/` stubs for CUDA/TensorRT headers |
| `deploy/get_sampleapp.sh`, `build_sysroot.sh`, `build_clayer.sh`, `build_deploy.sh`, `stage_board.sh` | SDK SampleApp copy, aarch64 sysroot, cyclonedds `_clayer`, the deploy cross-build, the board tree |
| `harness/run_closed_loop_iq9.py`, `iq9_runs.sh`, `board_quiet.sh`, `board_run.sh` | one closed-loop run on the board, the run batches (unloaded, CPU saturated, CPU arm, diagnostic, band drop, late-start settings), the quiet-board guard, the sonic_bench runner |
| `gate/to_gate_npz.py` | IQ-9075 outputs -> the input of the public `sonic/gate/score_gate.py` |

Reused unchanged from [sonic/](../../sonic/README.md): `harness/sonic_probe.{hpp,patch}`, `harness/latefix.patch`,
`harness/sim_driver.py`, `harness/mkref.py`, `gate/make_refsets.py`, `gate/score_gate.py`, `scoring/`, `make_sonic_table.py`.

## Credit and license

GEAR-SONIC and `g1_deploy_onnx_ref` are NVIDIA's work ([NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
Apache License 2.0; weights under the NVIDIA Open Model License). Nothing of theirs is vendored here; the scripts download or
copy it from your checkout. `deploy/shim/` replaces three upstream headers with QNN implementations of the same classes and
is distributed under the Apache License 2.0 (see [NOTICE](../../NOTICE)). Qualcomm's QAIRT/QNN SDK, its SampleApp sources and
the Hexagon libraries come from your own Qualcomm download under Qualcomm's license.
