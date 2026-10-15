# GR00T on the Qualcomm IQ-9075 (Hexagon v73 NPU)

The scripts to run GR00T N1.5 on the IQ-9075 yourself: the fp16 port of all 7 GR00T graphs to the Hexagon NPU, the
whole-policy speed loop, the closed-loop accuracy run against an RTX 5090 reference, int8 / int16 quantization builds and
gates, a SmoothQuant follow-up, and a hand-coded Hexagon int-vs-float microbench. **No IQ-9075 results are published in
this repo** -- run it and see.

The GEAR-SONIC whole-body controller (encoder/decoder on one NSP, split planner on the other, upstream's deploy cross-built
with a QNN shim, the MuJoCo closed loop) has its own guide: [iq9/sonic/](sonic/README.md).

**Heads-up: these are the original research scripts, packaged as they ran.** The NVIDIA side of the repo was rebuilt into
clean harnesses. This side is the original scripts (`work/`) with private paths swapped for environment variables. They
assume the directory layout below, talk to the board over ssh, and a few steps were driven by hand. Read each script's
docstring before running it.

### What the folder names mean

| Name | What it is |
|---|---|
| `work/closed_loop_m2/` | board transport: drives the board's resident orchestrator over ssh (`iq9_board.py`) and reproduces GR00T's preprocessing into board inputs (`iq9_prep.py`) |
| `work/closed_loop_m3/` | closed-loop driver + servers for the **base** checkpoint (zero-shot): 5090 / iq9 servers with an injected noise bank, the RoboCasa client, scene-determinism probes, the paired-rollout scorer `finalize_m3.py` |
| `work/closed_loop_m4/` | **fine-tuned** checkpoint: export + build + deploy (`build/`) and the closed-loop driver `m4_run.py` with its iq9 server and transport |
| `work/closed_loop_m5/` | scoring of a fine-tuned fp16 run vs the 5090 reference (`finalize_m5.py`) |
| `work/closed_loop_m7/` | int8 / int16 quantized builds (body and action head), per-graph gates, closed-loop arms |
| `work/closed_loop_m8/` | SmoothQuant variants of the LLM graph + their gate |
| `work/optionB/` | the resident NPU loop (`src/resident_orch.cpp`, board-side drivers) and its numpy host glue; `ah_fix/` = action-head fixes; `cfg_aaf0/` = HTP configs with `advanced_activation_fusion` off |
| `work/qnn/` | HTP configs for the first fp16 builds (`ah/` = action head, `w16/` = 16-bit weight variants) |
| `47` (`cfg47_*`, `ext47_*`, graph names `*_47`) | the rebuilt action-head / vlln graphs; the suffix is baked into those context binaries' graph names, so the drivers address them by it (`47b` = an O2 variant of the same build) |
| `gframe` | `GFRAME`, the orchestrator command that runs one whole policy call in-process (see the note in section 2) |
| `derank_gqa.py` | rewrites the LLM's rank-5 grouped-query-attention pattern at rank 4 for QNN |

`m1` and `m6` are not included.

## What you need

| Thing | Version we used | Where it comes from |
|---|---|---|
| Qualcomm IQ-9075 EVK (Dragonwing, Hexagon v73 NSP), Yocto image with fastrpc | -- | Qualcomm / Thundercomm |
| QAIRT / QNN SDK (x86 converter + board runtime) | **2.47.0.260601** | [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) or the [Qualcomm Software Center](https://softwarecenter.qualcomm.com) (Qualcomm account + license acceptance; not redistributed here) |
| Hexagon SDK (microbench only) | **6.6.0.0**, Hexagon tools 19.0.07 | [Hexagon NPU SDK](https://www.qualcomm.com/developer/software/hexagon-npu-sdk) via [Qualcomm Package Manager](https://qpm.qualcomm.com) (Qualcomm account + license acceptance) |
| Python on the board with numpy (+ OpenBLAS for the host-glue GEMM) | Python 3.12 | your choice; we used a self-contained Python tree |
| aarch64 cross compiler on the x86 host (`aarch64-linux-gnu-g++-11`) | 11 | your distro |
| An x86 host with the repo's GR00T + RoboCasa setup | -- | [docs/SETUP.md](../docs/SETUP.md) (`setup/setup_x86.sh`) |

**Every path comes from `setup/env.sh`.** The x86 side uses `GVRB_WORK`, `GROOT_REPO`, `GROOT_MODEL`, `GROOT_FT_MODEL`,
`GROOT_ONNX`, `GROOT_PY`, `ROBOCASA_PY` and `GROOT_SHIM` exactly as the rest of the repo does. The IQ-9075 adds:

```bash
source setup/env.sh
export QAIRT_SDK_ROOT=/path/to/qairt              # x86: contains 2.47.0.260601/
export IQ9_HOST=iq9                               # ssh host of the board (key auth); default "iq9"
export IQ9_WORK=/path/on/board/gvrb               # board: work dir (context bins, io, resident orchestrator)
export IQ9_QNN_ROOT=/path/on/board/qnn            # board: QNN runtime: bin/, lib/, and skel/ = the v73 skel libs (ln -s lib/hexagon-v73/unsigned skel)
export IQ9_PY_ROOT=/path/on/board/python          # board: Python tree
export GVRB_NOISE_BANK=$GVRB_WORK/noise_bank.npy  # optional; the closed_loop_m7 scripts default to this path
bash iq9/link_work.sh                             # symlinks iq9/work/* into $GVRB_WORK, where the scripts expect them
```

Because of `link_work.sh` the scripts run from your checkout and write their outputs next to themselves (build dirs,
`runs/`, logs); `.gitignore` keeps those out of git. **Never SIGKILL a process that holds the NSP** (`qnn-net-run`,
`resident_orch`): a killed DSP session can leave the NPU degraded (much slower, plausible-looking numbers) until reboot.
Every script here stops the board side with a clean `QUIT` / SIGINT.

**Order of work:** 1 (port) -> 2 (speed) -> 3 (fp16 closed loop) -> 4 (quantization, optional). Sections 5 and 6 are
independent.

## 1. Port: ONNX -> Hexagon fp16 context binaries

1. Export the fp16 ONNX graphs on the x86 host: `bash speed/export_onnx.sh` (same graphs the GPUs use, in `$GROOT_ONNX`).
2. The LLM's grouped-query attention exports rank-5 tensors that QNN's HTP validator rejects. Rewrite them at rank 4
   (the script checks the rewrite numerically against the original): `python iq9/work/derank_gqa.py <llm_fp16.onnx> <out.onnx>`.
3. Convert + compile each graph: `bash iq9/work/optionB/ah_fix/build_one.sh <onnx> <graph_name> [-s NAME DIMS]...`
   (`qairt-converter` -> DLC -> `qnn-context-binary-generator`, fp16_relaxed_precision, vtcm 8 MB, O3, v73).
   `iq9/work/optionB/build_groot_aaf0.sh` rebuilds a DLC with `advanced_activation_fusion: false`, the setting we used
   for the fp16 ViT on v73. HTP configs for every bin we built are in `iq9/work/qnn/` and `iq9/work/optionB/cfg_aaf0/`.
4. The action head has an embodiment-indexed (MoE-style) gathered-weight matmul, which can be mishandled by HTP fp16 in
   some QNN versions; the bake step avoids it -- check with the per-graph gate. `iq9/work/optionB/ah_fix/bake_eid.py`
   bakes the GR1 embodiment id (24) into a static matmul, `build_concatfree.py` removes the concat, `splice_and_ref.py`
   splices and makes the host references.
5. Extract the host-side glue constants once (mlp1 projector, embedding table, position embeddings) with
   `iq9/work/optionB/src/extract_glue_constants.py`; capture board inputs and host references with `groot_capture_ref.py`
   and `host_gate_graphs.py`.
6. Fine-tuned checkpoint (needed for section 3): `iq9/work/closed_loop_m4/build/export_ft.py` (ONNX of the graphs that
   change), `m4_build.sh` per graph (`run_heavy.sh` does ViT, vlln, DiT), `m4_host_ref.py` (host reference), then
   `deploy.sh` copies bins, glue and references to `$IQ9_HOST:$IQ9_WORK/optionB/{ctx_ft,glue_ft,io_ft}`.

## 2. Speed

- **Per-stage accelerator time** (ViT, LLM, one DiT step): on the board, `bash iq9/speed/stage_time_qnn.sh` -> JSON.
- **Whole policy call**: copy `Utils/ Log/ WrapperUtils/ PAL/ SampleApp.hpp QnnTypeMacros.hpp` from
  `$QAIRT_SDK_ROOT/2.47.0.260601/examples/QNN/SampleApp/SampleApp/src` into `iq9/work/optionB/src/` (Qualcomm-licensed;
  `.gitignore` keeps them out of commits) (next to `resident_orch.cpp`),
  cross-compile with `bash iq9/work/optionB/resident/build.sh` (-> `optionB/resident/build/resident_orch`), and copy it to
  `$IQ9_WORK/optionB/resident/` on the board. The board also needs the 7 context bins in `$IQ9_WORK/optionB/ctx_fix/`,
  the glue constants in `$IQ9_WORK/optionB/glue_constants/` and the inputs in `$IQ9_WORK/optionB/io/`. On the board,
  `optionB/src/resident_loop_groot.py` runs the policy through per-graph `EXEC` calls with the numpy host glue;
  `optionB/resident/resident_gframe_loop.py` times N in-process `GFRAME` frames, gates the action against the bf16
  golden, runs a zeroed-image null arm, and writes a JSON record.
  **The loops read pre-transformed inputs** (pixel_values, input_ids, masks made by the model's transforms on a host):
  image/text preprocessing is not in their time (the GPU harness in `speed/` includes it).

> **`resident_orch` commands:** `EXEC` (one graph, file-based, for debugging), `GFRAME` (one whole policy call
> in-process: all graphs plus the host glue, which the loops and the closed-loop board transports use), `PING`, `QUIT`.
> `GFRAME` needs `--glue <dir>` (the constants from `optionB/src/extract_glue_constants.py`, copied to the board;
> default `/opt/gvrb/optionB/glue_constants`) and `--blas <libopenblas.so>` (it dlopens OpenBLAS for one small GEMM;
> the copy inside the board's torch install works), optionally `--blas_threads N`.

## 3. Accuracy (closed loop, fp16)

Same task, seeds and noise bank as the GPU runs ([docs/ACCURACY.md](../docs/ACCURACY.md)). The simulator runs on the x86
host; the policy server on the host runs GR00T's transforms and drives the board's resident loop over ssh.

The **fine-tuned fp16 closed loop** (comparable with the GPU accuracy runs) is `closed_loop_m4/m4_run.py`, with the bins
from step 1.6:

```bash
$GROOT_PY accuracy/gen_noise.py --n_episodes 20 --seed_a 5000 --out_a $GVRB_WORK/noise_bank.npy --only_a   # once, as in docs/ACCURACY.md
$GROOT_PY iq9/work/closed_loop_m4/m4_run.py --backend iq9 --noise_bank $GVRB_WORK/noise_bank.npy \
    --n_episodes 20 --work $GVRB_WORK/runs/iq9_work --out_dir $GVRB_WORK/runs/iq9_fp16
```

`m4_run.py` starts `m4_server_iq9.py` with `$GROOT_PY` and the RoboCasa client with `$ROBOCASA_PY`; `--backend 5090`
runs the same driver with a local-GPU bf16 server as the reference. `closed_loop_m3/m3_run.py` is the same driver for the
base (zero-shot) checkpoint; `finalize_m3.py` scores its four paired arms (determinism control, noise envelope, fp16
effect, zeroed-image null).

Score your runs against a 5090 reference run (for example `results/accuracy/runs/rtx5090_bf16_reference`) without a board:
`python iq9/accuracy/score_iq9_runs.py --ref <5090 run> --run fp16=<your iq9 run> [--run int8_head=...] [--out scores.json]`.

## 4. int8 / int16 quantization

Scripts in `iq9/work/closed_loop_m7/`:

| Step | Script |
|---|---|
| calibration set from real closed-loop observations | `capture_obs_server.py`, `m7_capture_run.py`, `build_calib.py` |
| quantized builds (W8A8 per-tensor min-max / percentile, W8A16, W16A16, per-row weights; body and head) | `build_int8*.sh`, `build_a16*.sh`, `build_w16a16*.sh`, `build_head_pctl.sh`, `build_pctl_batch.sh`, `chain_prhead.sh` |
| per-graph cosine gates vs fp16 on a real frame (pass: cosine > 0.95) | `gate_prep.py`, `gate_run.py`, `gate_body_variants.py`, `gate_w16a16*.py`, `gate_a16pr_body.py` (write `int8_build/*gate*.json`) |
| closed-loop arms on the board | `m7_run.py`, `m7_server_iq9.py`, `iq9_board_m7.py`, `campaign*.sh` |
| finalize the record | `finalize_m7.py`, then `finalize_final.py <record.json>` to merge every arm's metrics |

SmoothQuant follow-up: `iq9/work/closed_loop_m8/` (`convert_llm.sh`, `build_m8.sh`, `chain_builds.sh`, `gate_m8.py`;
configs in `iq9/work/closed_loop_m8/cfg/`; `gate_m8.py` writes `m8_pergraph_gate.json`).

Then `$GROOT_PY iq9/int8_int16/quant_views.py --ref <5090 run> --fp16 <iq9 fp16 run> --arm int8_head=<run> --arm
int16_head=<run> [--gate int8=int8_build/pergraph_gate.json ...]` turns your runs into the "does closed-loop behaviour
change?" views: closed-loop success, per-joint action error of the first planned chunk, a joint trajectory, and your
per-graph cosines.

## 5. Hexagon int-vs-float microbench

`iq9/hexagon_microbench/`: `hmx_gemm.c` (fp16 HMX matrix-engine GEMM, unsigned-PD cDSP) and `hvx_fast.c` (uint8 HVX
FAST-corner loop). Build both into a Hexagon SDK cDSP skeleton for v73 (the SDK's `calculator` example is a fine host;
it is Qualcomm-licensed and not included). Then

```bash
python iq9/hexagon_microbench/disasm_mix.py calculator_hmx.o hmx_gemm mxclracc --objdump <sdk>/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
python iq9/hexagon_microbench/disasm_mix.py calculator_imp.o hvx_fast_rows vcmp --objdump ...
```

prints the hot loop and its FP / INT / MEM instruction mix. `disasm_mix.py --text <saved objdump listing>` re-classifies a
saved disassembly without the SDK.

## 6. Per-op profiling (matmul vs glue)

`iq9/profile/qnn_op_share.py` parses `qnn-profile-viewer` output from a `--profiling_level detailed` run into a matmul vs
glue share (QNN runs every op separately, so this is an unfused anatomy). It re-implements an interactive parse and has not
been re-run for this release; check its output against the viewer's raw listing.
