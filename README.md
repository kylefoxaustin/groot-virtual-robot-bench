# GR00T, measured: a virtual robot on RTX 5090, Jetson AGX Thor and Jetson AGX Orin

**So you want to see how well GR00T works? You've found the right repo.**

NVIDIA's GR00T N1.5 is a robot *policy*: it looks at a camera image, reads an instruction, and outputs joint commands.
We put it in charge of a simulated humanoid and measured two things on three NVIDIA machines: **how fast it decides**
and **whether it still decides the same way**. Every script we ran is here, along with the raw data and a checker that
rebuilds the published numbers from it. There are also scripts for the Qualcomm IQ-9075, so you can measure that one
yourself (we publish no IQ-9075 results).

## The headline

<!-- results:start -->
| Machine | Whole policy call (MEASURED) | Closed loop: success, 20 seeds | Same outcome as the RTX 5090 |
|---|---|---|---|
| RTX 5090 | 13.65 ms = **73.3 Hz** (full TensorRT) | 9/20 (bf16, the reference) | -- |
| Jetson AGX Thor | 48.27 ms = **20.7 Hz** | 10/20 fp16 · 8/20 bf16 · 10/20 fp8 | **19/20 fp16** · 19/20 bf16 · 17/20 fp8 |
| Jetson AGX Orin 64 GB | 88.01 ms = **11.4 Hz** | 9/20 fp16 | **20/20 fp16** |

Speed = NVIDIA's full TensorRT path, fp16 engines, image + instruction in, action chunk out. Closed-loop runs use PyTorch on each board, so their latencies are not the speed column. At n = 20 every success rate's 95% interval is about ±20 points: both Jetsons, at every precision, are statistically indistinguishable from the 5090 reference. The Thor fp16 run is from 2026-10-12 on a newer PyTorch stack; the other closed-loop runs are from 2026-09-18 ([RESULTS.md §5](RESULTS.md#5-accuracy-closed-loop-task-success-across-boards-measured-2026-09-18-thor-fp16-2026-10-12)).
<!-- results:end -->

Every number, with how it was measured: [RESULTS.md](RESULTS.md).

## Meet the virtual robot

A **Fourier GR1** humanoid (two arms, two hands, a waist) in **RoboCasa** (a MuJoCo simulator), told to *pick up the
milk, put it in the microwave and close the door*. A few numbers decide everything:

- **20 Hz**: the arm takes a new joint command every 50 ms.
- **16 steps**: each GR00T decision returns a 16-step *action chunk*, i.e. 800 ms of motion.
- So a board doesn't need to decide 20 times a second. It needs to finish the **next** decision before the current chunk
  runs out. "Hz" in our speed tables means *decisions per second*, not commands per second.
- **4 denoise steps**: GR00T plans each chunk with a small diffusion model, refined 4 times by default. More steps cost
  time linearly and shrink the difference from a 64-step reference only about 1/N -- that is solver error, not task
  success ([RESULTS.md §2](RESULTS.md#2-denoise-steps-cost-vs-solver-deviation-rtx-5090-measured)).

The speed runs time the base GR00T N1.5 model on one recorded frame of NVIDIA's simulated GR1 demo; the closed loop runs
the RoboCasa post-trained model on the milk task.

## Words we use

- **Policy call**: one full decision -- image + instruction in, action chunk out. Our speed numbers time exactly this.
- **ViT / LLM / DiT**: GR00T's three big parts -- the vision encoder (camera image to tokens), the language model, and
  the diffusion transformer that plans the action chunk (run once per denoise step). Four small modules glue them together.
- **Eager / hybrid / full TensorRT**: plain PyTorch; only ViT, LLM and DiT compiled with TensorRT; all 7 modules
  compiled (NVIDIA's own deployment path).
- **Closed loop**: the policy drives the simulator and sees the result of its own actions, for a whole episode.
- **Seed**: one reproducible episode (scene layout + diffusion noise). We run seeds 0-19.
- **Success** vs **same outcome**: success = the task got done; same outcome = on how many of the 20 seeds the board's
  success or failure matched the RTX 5090's.
- **bf16 / fp16 / fp8**: number formats for the arithmetic, from 16-bit down to 8-bit floating point.
- **MEASURED / DERIVED / SOURCED**: we ran it / we computed it from measurements / someone else published it.

## Pick your adventure

| I have... | Do this | You get | Time |
|---|---|---|---|
| **Just a laptop** | `git clone https://github.com/kylefoxaustin/groot-virtual-robot-bench.git && cd groot-virtual-robot-bench && python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy && /tmp/v/bin/python tools/check_results.py` | our speed, denoise, closed-loop and DDR numbers rebuilt from the shipped raw data; it prints `OK` or what differs. The numbers themselves are in [RESULTS.md](RESULTS.md) | ~1 min |
| **An x86 Linux PC with an NVIDIA GPU** | [docs/SETUP.md](docs/SETUP.md), then [docs/SPEED.md](docs/SPEED.md) | eager vs hybrid vs full-TensorRT policy-call speed on your GPU, two engine builds | setup: ~22 GB of models and sim assets plus several GB of pip wheels |
| ...the same PC, and patience | [docs/SETUP.md](docs/SETUP.md), then [docs/ACCURACY.md](docs/ACCURACY.md) | 20 closed-loop episodes on your GPU, seed by seed against our shipped RTX 5090 run | 20 episodes × 720 sim steps; GPU time is only ~1 min of it (900 calls × ~60 ms on the 5090), the simulator the rest |
| **A Jetson AGX Thor or Orin** | the same two docs; each has a Jetson section | the same speed run on the board, and the board as closed-loop policy server (the simulator stays on an x86 PC) | |
| **A Qualcomm IQ-9075** | [iq9/README.md](iq9/README.md) (needs Qualcomm's QAIRT SDK) | scripts to port GR00T to the Hexagon NPU and measure speed, closed-loop behaviour and int8/int16 yourself | |
| **A question about where the time goes** | [profiling/README.md](profiling/README.md) | matmul vs glue, integer vs floating-point instructions, DDR bandwidth (mostly Jetson) | |

Full map of every benchmark, its script, its result file and how sure we are: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## What you'll need to download

Nothing proprietary ships in this repo. You only need the rows for the hardware you have; `setup/` fetches most of it.

| You need it for | What | Version we used | Where to get it | Account / license? |
|---|---|---|---|---|
| Speed benchmarks | GR00T N1.5 base model | `nvidia/GR00T-N1.5-3B` @ pinned rev | [Hugging Face](https://huggingface.co/nvidia/GR00T-N1.5-3B), fetched by `bash setup/fetch_models.sh` | No account; NVIDIA One-Way Noncommercial model license |
| Closed-loop accuracy | GR00T N1.5 RoboCasa post-trained model | `youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain` @ pinned rev | [Hugging Face](https://huggingface.co/youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain), fetched by `bash setup/fetch_models.sh ft` | No account; derived from GR00T N1.5, same license |
| Everything on a GPU | Isaac-GR00T, robosuite, RoboCasa GR1 code | pinned commits | Cloned by `bash setup/fetch_upstream.sh` | Free (Apache-2.0 / MIT) |
| x86 GPU | NVIDIA driver (CUDA 12.8 and 13 capable) | 580.173.02 (any R580+) | [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/) | Free |
| x86 GPU | TensorRT | 11.2.1.2 | `pip install` from [PyPI](https://pypi.org/project/tensorrt/) (done by `setup/setup_x86.sh`) | Free |
| Jetson AGX Thor | JetPack (includes CUDA + TensorRT) | JetPack 7.2.1 | [JetPack](https://developer.nvidia.com/embedded/jetpack) via [SDK Manager](https://developer.nvidia.com/sdk-manager) | Free NVIDIA developer account |
| Jetson AGX Thor | PyTorch container for the speed runs | `nvcr.io/nvidia/pytorch:25.08-py3` | [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch); `setup/thor_container.sh` pulls it | Free |
| Jetson AGX Thor | PyTorch for the closed-loop server | torch 2.14.0+cu130 (aarch64) | PyTorch's `https://download.pytorch.org/whl/cu130` index ([docs/SETUP.md](docs/SETUP.md)) | Free |
| Jetson AGX Orin | JetPack (includes CUDA + TensorRT 10.3) | JetPack 6 (L4T R36.4) | [JetPack](https://developer.nvidia.com/embedded/jetpack) via [SDK Manager](https://developer.nvidia.com/sdk-manager) | Free NVIDIA developer account |
| Jetson AGX Orin | PyTorch for Jetson wheels | torch 2.8.0 / torchvision 0.23.0 (CUDA 12.6) | [jetson-ai-lab index](https://pypi.jetson-ai-lab.io) or [NVIDIA's Jetson wheels](https://developer.download.nvidia.com/compute/redist/jp/) | Free |
| Profiling (int vs FP mix) | Nsight Compute (`ncu`) | ships with CUDA / JetPack | [Nsight Compute](https://developer.nvidia.com/nsight-compute) | Free |
| Qualcomm IQ-9075 | QAIRT / QNN SDK (converter + on-board runtime) | 2.47.0.260601 | [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk), or the [Qualcomm Software Center](https://softwarecenter.qualcomm.com) | Qualcomm account; accept Qualcomm's license |
| IQ-9075 Hexagon microbench only | Hexagon SDK (+ Hexagon tools 19.0.07) | 6.6.0.0 | [Hexagon NPU SDK](https://www.qualcomm.com/developer/software/hexagon-npu-sdk), installed with [Qualcomm Package Manager](https://qpm.qualcomm.com) ([getting started](https://docs.qualcomm.com/bundle/publicresource/80-77512-1/topics/hexagon-dsp-sdk-getting-started.md)) | Qualcomm account; license acceptance required before download |

Where each one plugs in (env vars, paths): [docs/SETUP.md](docs/SETUP.md) for NVIDIA, [iq9/README.md](iq9/README.md) for Qualcomm.

## FAQ (a.k.a. the fine print)

**Is the simulator real-time?** No. It pauses while the policy thinks, so closed-loop success measures *decision
quality*, not whether a board keeps up with a real arm.

**Why "whole policy call" and not engine benchmarks?** Engines timed alone and added up overstate the rate a robot sees:
by about a third even on NVIDIA's full TensorRT path. We report the whole call (image + instruction in, action chunk out),
with host-side work included. Camera capture and robot I/O are still extra.

**Where are the IQ-9075 numbers?** Not published here -- only the scripts. If you measure it, note that the resident NPU
loop reads pre-transformed inputs, so image/text preprocessing is not in its time while the GPU times include it.

**n = 20?** Twenty seeded episodes per backend: enough to rule out a collapse, not a 10-20 point degradation.

**Clocks, versions, build luck?** TensorRT builds vary a few percent from build to build, so we always build twice.
Versions are recorded in every result file, and clocks wherever we recorded them (the speed records do; a few profiling records don't). Yours will differ a little; tell us if they differ a lot.

**Does the checker change my files?** No. It regenerates everything into a temporary directory and compares.

**What's not here?** Model weights, NVIDIA's and Qualcomm's SDKs (see [What you'll need to download](#what-youll-need-to-download)), and other accelerators.

## Where things live

```
README.md   you are here          RESULTS.md  every published number (generated from results/)
docs/       setup, speed and accuracy how-tos, and BENCHMARKS.md (the map of every benchmark)
setup/      pinned versions, env.sh, per-platform setup, fetchers, the one Isaac-GR00T patch
speed/      ONNX export, engine builder, whole-call harness, denoise sweeps, arms-vs-cameras benches
accuracy/   closed-loop harness: noise bank, policy servers (incl. fp8), sim client, scoring
shims/      stand-ins for packages GR00T imports but never calls on our paths (see docs/SETUP.md)
profiling/  where the time goes: per-engine timing, matmul vs glue, int vs FP, DDR bandwidth, Thor DiT anomaly
iq9/        scripts for the Qualcomm IQ-9075: port, speed, closed loop, int8/int16, Hexagon microbench (no results)
results/    every published number as JSON + the raw data behind it (see results/README.md)
tools/      make_results_md.py (writes RESULTS.md + the headline above), check_results.py (rebuilds and compares)
```

## Licenses

Our code and results: Apache License 2.0 ([LICENSE](LICENSE)). **Model weights are not included** and come under
NVIDIA's model license (GR00T N1.5-3B: NVIDIA One-Way Noncommercial License) -- read it before use. robosuite and
RoboCasa-GR1 are MIT; Isaac-GR00T is Apache-2.0; RoboCasa's downloaded simulation assets carry their own licenses.
Qualcomm's SDKs are not included. See [NOTICE](NOTICE).
