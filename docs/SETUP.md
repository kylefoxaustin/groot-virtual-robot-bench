# Setup

You need this page for anything beyond `tools/check_results.py`. The x86 part alone is enough for the speed benchmarks
and for the closed loop on a single NVIDIA GPU; the Jetson parts only matter if you have one.

## What you need

| Machine | Role | Software we used |
|---|---|---|
| x86 Linux + one NVIDIA GPU (ours: RTX 5090, 32 GB) | speed benchmarks; ONNX export; **runs the simulator for every closed-loop run** (EGL rendering); can also be the closed-loop policy server | Ubuntu, Python 3.12 (policy) + 3.10 (sim), torch 2.9.0+cu128, TensorRT 11.2.1.2 (pip), a driver supporting CUDA 12.8 and 13 (R580+) |
| Jetson AGX Thor (optional) | speed (container) + closed-loop policy server | JetPack 7.2.1; speed in `nvcr.io/nvidia/pytorch:25.08-py3` (TensorRT 10.13); closed loop in a venv with torch 2.14.0+cu130 |
| Jetson AGX Orin 64 GB (optional) | speed + closed-loop policy server | JetPack 6 (L4T R36.4), system Python 3.10, torch 2.8.0 (Jetson wheel), TensorRT 10.3.0 |

- **Disk:** ~40 GB on the x86 host (models ~13 GB, RoboCasa assets ~9 GB, ONNX, engines, venvs), ~25 GB per Jetson.
- **GPU:** we only ran on a 32 GB RTX 5090 and did not measure peak memory or try other GPUs. The model has ~3B
  parameters (~6 GB of bf16 weights). If it works (or doesn't) on your card, please open an issue.
- **Python:** both `python3.12` and `python3.10`, each with `venv`. Ubuntu ships only one of them:
  `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12-venv python3.10-venv`. `setup_x86.sh` checks first.
- **Accounts:** none for the NVIDIA path. Both models download from Hugging Face without a token (they are not gated),
  but they are under NVIDIA's One-Way Noncommercial License -- read it.
- **Closed loop with a Jetson:** the Jetson must be reachable from the x86 host over ssh (key auth) and TCP port 5571 (humanoid);
  the drone's policy server listens on 127.0.0.1:47124 and is reached through an ssh tunnel ([drone/README.md](../drone/README.md)).

## Install

Everything large goes under `$GVRB_WORK` (default `~/gvrb`); every script reads its paths from `setup/env.sh`.
**Run `source setup/env.sh` in every new shell** before using the commands in these docs.

```bash
git clone https://github.com/kylefoxaustin/groot-virtual-robot-bench.git && cd groot-virtual-robot-bench
source setup/env.sh            # set GVRB_WORK etc. first if you want other locations
```

**x86 host:** `bash setup/setup_x86.sh`. It:

1. clones Isaac-GR00T, robosuite and RoboCasa-GR1 at the pinned commits (`setup/versions.env`) and applies our one patch
   (`setup/patches/isaac-gr00t-n15-export_onnx.patch`);
2. builds two venvs: `venv-groot` (Python 3.12: GR00T, TensorRT) and `venv-robocasa` (Python 3.10: the simulator);
3. installs flash-attn 2.8.3 straight from the prebuilt wheel on flash-attn's GitHub releases (torch 2.9 + CUDA 12 +
   Python 3.12), so no CUDA toolkit is needed. If that download fails it falls back to building from source, which needs
   `nvcc` (CUDA 12.8 toolkit), `ninja`, plenty of RAM (`MAX_JOBS=4` helps) and a long time;
4. downloads the simulator assets and both checkpoints, and checks the weights' sha256.

If `git-lfs` is missing the clone still works (`GIT_LFS_SKIP_SMUDGE=1`; the demo data the benchmarks use is not in LFS).
The `requirements-*.txt` files pin the packages that matter. The `freeze-*.txt` files are the complete `pip freeze` of the
environments behind the published numbers, kept for reference and not enforced. RoboCasa's own dependencies also pull a
full CUDA torch into the simulator venv.

**Jetson AGX Orin (JetPack 6):** install the Jetson torch 2.8.0 / torchvision 0.23.0 wheels for JetPack 6 (CUDA 12.6) --
NVIDIA's "PyTorch for Jetson" wheels or the jetson-ai-lab index -- then `GROOT_PY=python3 bash setup/setup_orin_jp6.sh`.

**Jetson AGX Thor (JetPack 7):** speed benchmarks run in the NGC container via `bash setup/thor_container.sh '<command>'`.
For the closed-loop server, make a venv with an aarch64 CUDA 13 torch build (we used torch 2.14.0+cu130 from PyTorch's
`https://download.pytorch.org/whl/cu130` index) plus GR00T N1.5's pins -- **transformers 4.51.3, tokenizers 0.21.4,
huggingface_hub 0.35.3**, the rest as in `setup/requirements-orin-jp6.txt`. A bf16 control run in this environment
reproduced the earlier Thor bf16 outcomes on all 20 seeds (2026-10-12). Then `bash setup/fetch_upstream.sh && bash setup/thor_container.sh 'bash setup/fetch_models.sh all'` (the container has
`huggingface_hub`; JetPack 7's system Python does not, and will not let pip install into it). The container needs docker with the
NVIDIA runtime, your user in the `docker` group, ~35 GB for the image, and network access (each start pip-installs GR00T's pins).

**Both Jetsons, for the closed loop:** Jetson has no flash-attn, so switch GR00T's vision-language backbone to PyTorch
SDPA attention (exact attention, different kernel): `$GROOT_PY accuracy/board_patch_sdpa.py --isaac $GROOT_REPO apply`
(`restore` undoes it).

**What is `shims/`?** GR00T N1.5 imports a few packages it never calls on our paths (pytorch3d; on Jetson also decord and
flash_attn) and that have no wheel for that platform. `setup/env.sh` puts tiny stand-ins from `shims/` on `PYTHONPATH`;
each raises an error if anything actually calls it.

Next: [docs/SPEED.md](SPEED.md) or [docs/ACCURACY.md](ACCURACY.md). For the Qualcomm IQ-9075 see
[iq9/README.md](../iq9/README.md); for the profiling scripts see [profiling/README.md](../profiling/README.md).
