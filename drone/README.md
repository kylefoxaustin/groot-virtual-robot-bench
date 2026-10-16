# GR00T flies a drone: same brain, new body

GR00T N1.5 (the rest of this repo) drives a humanoid. Its first half is a vision-language model: camera image +
instruction in, a 2048-number summary out. That half knows nothing about arms. We kept it **frozen** and trained a small
**action head** (1.35 M weights) to turn its summary into flight commands for a simulated quadrotor. Then we asked the
same questions as for the humanoid, on an RTX 5090, a Jetson AGX Thor and a Jetson AGX Orin 64 GB:

1. **How fast does it decide?** The whole decision, host work included (*full cost*).
2. **Does every machine make the same decisions?** An output gate against the RTX 5090.
3. **Does it get there?** 40 held-out flights, also while the drone keeps flying during each decision.
4. **What if the computer has other AI jobs?** Inspection, radar and route-planning stand-ins on the same GPU.

The numbers are in [RESULTS.md §9](../RESULTS.md#9-gr00t-as-a-drones-flight-brain-measured-2026-10-14--2026-10-15) and the
[README](../README.md#new-same-brain-new-body-gr00t-flies-a-simulated-drone). This page is how they were measured and how to run it.
IQ-9075: scripts only, in [iq9/](iq9/README.md).

## The task and the policy

**The simulator** ([sim/drone_sim.py](sim/drone_sim.py), MuJoCo 3.8.1): a quadrotor moved kinematically. Every 20 ms it
applies one body-frame velocity setpoint `[vx, vy, vz, yaw_rate]`; a real drone's flight controller would track those.
A forward camera renders 224 × 224 RGB. The scene has a checkered floor, a sky and four coloured pillars. The task:
*"fly to the red (green, blue, yellow) pillar"*. Reached = within 0.8 m of it horizontally, within 1,280 ticks (25.6 s).

**The policy** ([policy.py](policy.py)), one decision:

```
224 px frame + "fly to the red pillar" + state [x, y, z, yaw]
  -> GR00T's own input transforms (host CPU)
  -> ViT (TensorRT fp16) -> projector + embeddings (PyTorch, NVIDIA's deployment code, unmodified)
  -> LLM (TensorRT fp16, 296-token engine; the 289-token prompt is right-padded)
  -> vlln + vl_self_attention (GR00T's 4-layer vision-language attention block; TensorRT fp16, 289-token engine) -> mean
  -> action head (numpy on the host CPU)  -> 16 setpoints, 20 ms apart
```

No diffusion transformer: the drone head replaces GR00T's action head. The head was trained by behaviour cloning on
120 episodes of a scripted teacher (4,603 frames; [train/](train/)). It is in [assets/drone_head.npz](assets/), with the
normalisation statistics it was trained with. Every published number used exactly that file.

**Words used below.** *Lockstep, K = 4*: the simulator pauses for each decision, then flies the first 4 setpoints (80 ms).
*Delay d*: the drone keeps flying the previous setpoints for d ticks of 20 ms after the camera frame, then the new chunk
takes over. *Back-to-back*: the next decision starts as soon as the last one finishes, so d = the decision's own latency.
*Naive / aligned*: a late chunk starts at its first setpoint, or skips the d setpoints whose time has already passed.
*σ*: the spread (standard deviation) of each setpoint in the training data; deviations are given in σ.

## How it was measured

| Question | Script | Protocol |
|---|---|---|
| Full-cost rate | [run_fullcost.sh](run_fullcost.sh) → [drone_fulltrt.py](drone_fulltrt.py) `fullcost` | 3 runs × 64 decisions (+5 warm-up), each its own process, frame offsets 0/64/128 of the 256 gate frames. Raw frame + prompt + state → setpoints. Rate = 1000 / median of the run medians. |
| Output gate | [drone_fulltrt.py](drone_fulltrt.py) `gate` → [score_gate.py](score_gate.py) | The 256 gate frames ([assets/gate_set.npz](assets/)): the 4 setpoints a K = 4 loop flies, in vx / vz / yaw_rate, vs the RTX 5090's eager bf16 policy. PASS = RMS ≤ 0.07σ, per-frame p95 ≤ 0.20σ, a zero-image null run deviating ≥ 10× more than the real frames, and a planted fault (another frame's features) must fail. |
| Closed loop | [closed_loop.py](closed_loop.py) + [delay_loop.py](delay_loop.py) | Seeds 7000–7039. RTX 5090: simulator and policy in one process. Jetsons: the board serves decisions (`drone_fulltrt.py serve`), the simulator runs on the x86 machine and talks to it over an ssh tunnel. All rendering with Mesa's software rasteriser llvmpipe, which repeats bit for bit. |
| Other AI jobs | [contention/](contention/) | The nav policy back to back while stand-ins run in separate processes on the same GPU. Nav rate is binned over each co-tenant's own [start, end]; every window's decisions are gated. |

Jetsons ran in their maximum power mode (MAXN) with clocks pinned (`jetson_clocks`); the RTX 5090's clocks were not locked. Software per machine, engine checksums and
caveats: [results/drone/drone_setup_2026-10-15.json](../results/drone/drone_setup_2026-10-15.json).

**Three things we learned the hard way**, all handled by the scripts:

- **The renderer.** NVIDIA's EGL renders differ in the last bit of some pixels from process to process, and a closed loop turns that into
  different flights. Mesa's llvmpipe does not. [sim/render_check.py](sim/render_check.py) tells you if yours renders
  bit-identically to ours.
- **Host threads.** The head is a small numpy matmul; let BLAS spin up threads and they steal the cores GR00T's input
  transforms need. Every entry script sets `OPENBLAS_NUM_THREADS=1` before importing numpy.
- **Hybrid CPUs.** On a CPU with performance and efficiency cores (our RTX 5090 host), the decision takes 7.2 ms on a
  P-core and 8.6 ms on an E-core, and the OS moves it every few seconds. Pin with `taskset` if you want a stable number.

## Run it

**What you need:** the repo's GR00T setup for your machine ([docs/SETUP.md](../docs/SETUP.md); for an x86 PC
`bash setup/setup_x86.sh --policy-only` is enough), the base model (`nvidia/GR00T-N1.5-3B`, fetched by the setup), and
`source setup/env.sh` in every shell (it also sets the drone paths: `DRONE_WORK`, `DRONE_ENG`, `DRONE_RUNS`, ...).
Mesa's EGL for the software renderer: `sudo apt install libegl-mesa0` (Ubuntu ships it with the desktop).

### x86 Linux + NVIDIA GPU (ours: RTX 5090)

```bash
source setup/env.sh
bash drone/setup.sh                        # MuJoCo + PyOpenGL into the GR00T venv, asset check, ONNX export, 3 engines
bash drone/run_fullcost.sh mygpu rtx5090   # gate + 3 full-cost runs; prints PASS/FAIL and your rate next to ours
```

Closed loop, 10 of the 40 flights, then episode by episode against ours:

```bash
$GROOT_PY drone/closed_loop.py --out $DRONE_RUNS/cl --arm k4 --K 4 --episodes 0-9
python3 drone/compare_closed_loop.py $DRONE_RUNS/cl/k4_episodes.jsonl results/drone/raw/closed_loop/rtx5090_trt_np_K4_llvmpipe.jsonl
```

Other schedules: `--K 15` (lockstep, 15 setpoints), `--d 3 --dmode naive` (3 ticks of delay injected), `--d own`
(back-to-back at the measured latency). The published arms are `results/drone/raw/closed_loop/*.jsonl`; the file name says
which (`rtx5090_delay_trt_K15_d3_naive` = K 15, d 3, naive). On an RTX 5090 a flight takes 1–20 s; 40 flights about 5 minutes.

**What to expect.** With the same GPU, TensorRT version and renderer, flights match ours exactly (same outcome, decision
count and final distance). Another GPU or TensorRT version builds slightly different engines: most flights still end
the same way, a few take a different path. Compare reach and same-outcome counts then, not decision counts.

### Jetson AGX Orin (JetPack 6) and Jetson AGX Thor (JetPack 7)

**You need an x86 machine with an NVIDIA GPU once**, with the x86 setup above (`setup_x86.sh --policy-only`, ~20 GB): it
exports the ONNX graphs and the GR00T input step, which you copy to the board (the export reads NVIDIA's demo video, which
the Jetson images cannot decode). Same step as [docs/SPEED.md](../docs/SPEED.md#on-a-jetson):

```bash
bash speed/export_onnx.sh && $GROOT_PY speed/make_step0.py $GROOT_STEP_NPZ
rsync -a $GROOT_ONNX $GROOT_STEP_NPZ <jetson>:gvrb/        # into the board's $GVRB_WORK (default ~/gvrb)
```

**On the board**, after the repo's Jetson setup ([docs/SETUP.md](../docs/SETUP.md)). The drone needs only the base model:
Orin `GROOT_PY=python3 bash setup/setup_orin_jp6.sh --policy-only`; Thor `bash setup/fetch_upstream.sh && bash setup/thor_container.sh 'bash setup/fetch_models.sh base'`
(Thor runs everything GPU-side in NVIDIA's PyTorch container: you need docker with the NVIDIA runtime, your user in the `docker`
group, ~35 GB for the image, and network access, because each start pip-installs GR00T's pins). Pin the clocks first
(`sudo nvpmodel -m 0 && sudo jetson_clocks`: maximum power mode, MAXN, with pinned clocks; a reboot undoes it, and an
unpinned Thor ran about 23% slower in our test). Setup with engines takes about 2 minutes on either board:

| Step | Orin (native) | Thor (NVIDIA's PyTorch container) |
|---|---|---|
| setup + engines | `bash drone/setup.sh` | `bash setup/thor_container.sh 'bash drone/setup.sh'` |
| rate + gate | `bash drone/run_fullcost.sh orin orin` | `bash setup/thor_container.sh 'bash drone/run_fullcost.sh thor thor'` |
| policy server | `python3 drone/drone_fulltrt.py serve --out $DRONE_RUNS/serve` | `THOR_DOCKER_ARGS="--network host" bash setup/thor_container.sh 'python drone/drone_fulltrt.py serve --out $DRONE_RUNS/serve'` |

Keep the single quotes around Thor commands: the variables inside expand in the container, where `setup/thor_container.sh`
sets them to the same paths.

**Flights.** On a Jetson the simulator runs in a separate process from the policy: the board serves decisions
(port 47124 on 127.0.0.1), and `closed_loop.py --policy` flies against it. (Running both in one process crashes on Jetson:
Mesa's EGL and the GPU stack collide.) Our published Jetson flights ran the simulator on the x86 machine, through an ssh tunnel:

```bash
ssh -f -N -L 47124:127.0.0.1:47124 <jetson>                      # on the x86 machine
B=orin                                                           # or thor
$GROOT_PY drone/closed_loop.py --policy 127.0.0.1:47124 --out $DRONE_RUNS/cl --arm ${B}_k4 --K 4 --episodes 0-9
$GROOT_PY drone/closed_loop.py --policy 127.0.0.1:47124 --out $DRONE_RUNS/cl --arm ${B}_own --d own --episodes 0-9
python3 drone/compare_closed_loop.py $DRONE_RUNS/cl/${B}_k4_episodes.jsonl results/drone/raw/closed_loop/${B}_K4_lockstep.jsonl
python3 drone/compare_closed_loop.py $DRONE_RUNS/cl/${B}_own_episodes.jsonl results/drone/raw/closed_loop/${B}_naive_backtoback.jsonl
python3 drone/server_ctl.py quit 127.0.0.1:47124                 # stops the server cleanly
```

`--d own` is the back-to-back run at the board's own latency.
No x86 machine? Run the same client on the board's own CPU in a small venv: `python3 -m venv ~/venv-sim &&
~/venv-sim/bin/pip install -r drone/requirements-sim.txt` (Ubuntu needs `sudo apt install python3-venv libegl-mesa0`), then
`~/venv-sim/bin/python drone/closed_loop.py --policy 127.0.0.1:47124 ...`. The ARM build of llvmpipe renders slightly
differently from x86 (check with [sim/render_check.py](sim/render_check.py)), so flights will not match ours one for one;
compare reach and same-outcome counts.

**What we saw reinstalling from scratch** (2026-10-16; MEASURED during our install test, not part of the published results, no raw data in `results/`): the Orin rebuilt bit-identical engines, so its gate deviation from our
Orin run was 0.0000σ and 10 of 10 flights matched ours exactly, lockstep and back-to-back. Thor's rebuilt engines differ
slightly from ours (0.003σ), so its flights took slightly different paths: 9 of 10 (lockstep) and 10 of 10 (back-to-back) ended
the same way as ours. Both gates passed. Our full-cost numbers were reproduced within 1.1% (Orin) and 2.6% (Thor).

### Other AI jobs on the same computer (contention)

```bash
$GROOT_PY drone/contention/make_standins.py              # seeded stand-ins into $DRONE_STANDINS (CPU, ~10 s)
bash drone/contention/setup_workers_x86.sh               # x86 only: a venv with TensorRT 10.16.1 for the stand-ins
bash drone/contention/build_engines.sh x86               # thor | orin on a Jetson (uses JetPack's trtexec)
bash drone/contention/run_contention.sh scan my_scan     # 75 s: nav alone, then 5 inspection bursts
python3 drone/contention/score_run.py $DRONE_RUNS/contention/my_scan --compare e1_scan_rtx5090
```

On a Jetson: make the stand-ins on the x86 machine and copy `$DRONE_STANDINS/*.onnx` over (like the ONNX export); the
workers use JetPack's own python3 + TensorRT (plus `cuda-python`: `pip install cuda-python` if
`python3 -c "from cuda.bindings import runtime"` fails). Compare with `e1_scan_orin` / `e1_scan_thor`. On Thor add `NAV_IN_THOR_CONTAINER=1` so the nav policy runs in the container
while the workers stay native, as in our runs. Modes: `scan` (the 50 MP inspection stand-in, 190 tiles of 512 × 512), `radar` (fusion at 20 Hz), `radarflat` (fusion back
to back), `all` (radar + route planning throughout, plus the inspection bursts). Published runs to compare with:
`ls results/drone/raw/contention/`. `make_standins.py` prints the sha256 of every file; with torch 2.9.0 /
torchvision 0.24.0 they equal [results/drone/standins_MANIFEST.json](../results/drone/standins_MANIFEST.json).
The stand-ins have random weights: they load the computer like the real jobs would, they do not do the jobs.

**Never SIGKILL a process that holds the GPU.** Every script here ends on its own; the policy server stops on a `quit`
request (`compare_closed_loop.py` is read-only).

## Rebuild our numbers (no GPU)

```bash
python3 drone/make_drone_table.py /tmp/drone_table.json   # every rate, gate value, reach count and test, from results/drone/raw/
python3 tools/check_results.py                            # also compares it with the committed table, RESULTS.md and the README
```

## Retrain the head (optional)

`python3 drone/train/collect_data.py DIR 120` flies the scripted teacher (seeds 1000–1119), then
`$GROOT_PY drone/train/train_head.py DIR` extracts the frozen GR00T features (eager bf16) and trains the head (~15 min on
an RTX 5090). Our training split was not seeded, so your head will differ from ours and so will its flights. Use
[assets/drone_head.npz](assets/) to compare with our numbers.

## Files

| Path | What |
|---|---|
| `policy.py`, `drone_head_np.py` | the decision path (TensorRT or eager backbone + numpy head) |
| `drone_fulltrt.py` | `gate`, `fullcost`, `loop` (for contention) and `serve` (policy server for a remote simulator) |
| `closed_loop.py`, `delay_loop.py`, `compare_closed_loop.py`, `server_ctl.py` | flights (local or against a server), the delay model, comparison with ours |
| `score_gate.py`, `score_fullcost.py` | score your gate and full-cost runs, next to ours |
| `sim/` | the simulator, the scene, the render check |
| `contention/` | stand-in generator, engine builder, workers, orchestrator, scorer |
| `train/` | data collection and head training |
| `assets/` | the trained head, the 256 gate frames with the reference actions, their manifest and license |
| `make_drone_table.py`, `drone_metrics.py` | rebuild `results/drone/drone_table_2026-10-15.json` from `results/drone/raw/` |
| `setup.sh`, `env.sh`, `requirements-drone.txt`, `requirements-sim.txt`, `run_fullcost.sh` | setup and the one-command full-cost run |
| `iq9/` | IQ-9075 scripts (no results) |

## Credit and license

GR00T N1.5 is NVIDIA's ([Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), weights
[nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) under the NVIDIA One-Way Noncommercial License). Our
code is Apache-2.0. `assets/drone_head.npz` and the reference actions in `assets/gate_set.npz` were made from GR00T N1.5, so
they are distributed under that NVIDIA license (non-commercial use only; a copy is in `assets/`). MuJoCo is Apache-2.0.
