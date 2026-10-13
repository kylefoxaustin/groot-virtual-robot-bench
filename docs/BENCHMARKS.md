# Every benchmark, where it lives, and how sure you can be

One row per measurement behind our GR00T results. **Status** tells you what this repo can do with it:

- **regenerates** -- the raw data is in `results/`, and `tools/check_results.py` rebuilds the published numbers from it
  (no GPU needed). Re-run the script on your hardware to get your own raw data.
- **re-run 2026-10-12** -- the original script was not preserved; we re-implemented it from the record's method and re-ran
  it to confirm the result.
- **published record** -- the script that produced it is here and runnable, but the raw data is not, so the number is the
  record as published (not regenerated here).
- **scripts only** -- the Qualcomm IQ-9075 rows: everything needed to measure it is here, but no IQ-9075 result is published.
- **method script** -- a re-implementation of one step from the record's documented method, not re-run for this release.
- **DERIVED** -- no measurement of its own: arithmetic on the measured rows it names.

## Speed

| Measurement | Boards | Script | Result | Status |
|---|---|---|---|---|
| Whole policy call: eager vs hybrid vs NVIDIA's full TensorRT, same session, 2 builds | 5090, Thor, Orin | `speed/run_board.sh` → `speed/e2e_eager_hybrid_fulltrt.py` | `results/speed/groot_*_fulltrt_e2e_2026-10-12.json` | regenerates |
| Engines timed alone, same session (ViT + LLM + 4×DiT and all 7) | 5090, Thor, Orin | same harness (`standalone_engine_ms`) | same records | regenerates |
| Engines timed alone, 3-6 builds per board, vs NVIDIA's published Orin table | 5090, Thor, Orin | `profiling/compiled_fleet/` | `results/profiling/groot_compiled_fleet_2026-08-27.json`, `results/speed/engine_stage_times_2026-08-27.json` | published record |
| Denoise-steps sweep 1→32 + 64-step reference, full TensorRT | 5090 | `speed/run_denoise_sweep.sh` → `speed/denoise_sweep_fulltrt.py` | `results/speed/n15_5090_denoise_fulltrt_2026-10-12.json` | regenerates |
| Stage split at 4 steps, full TensorRT | 5090 | same | `results/speed/n15_5090_stagetime_fulltrt_2026-10-12.json` | regenerates |
| Denoise sweep + stage split, PyTorch eager | 5090 | `speed/denoise_sweep_eager.py`, `speed/stage_time_eager.py` | `results/speed/n15_5090_*_eager_2026-09-23.json` | published record |
| 2 arms vs 4 arms (action encoder + decoder) | 5090, Thor, Orin | `speed/arms_2v4/` | `results/speed/arms_2v4_2026-10-11.json` | published record (raw timings for 5090 and Orin in `results/speed/raw/arms_2v4_*.json`; not rebuilt by the checker) |
| Arms vs cameras (action-head modules, ViT 1/2/4 views, LLM 256/512/1024 tokens) | 5090 | `speed/multiarm/camera_token_scaling.py` | `results/speed/multiarm_scaling_rerun_2026-10-12.json` (vs `multiarm_scaling_2026-10-10.json`) | re-run 2026-10-12 |
| IQ-9075 per-stage accelerator time (ViT, LLM, DiT step) | IQ-9075 | `iq9/speed/stage_time_qnn.sh` | -- | scripts only (no published IQ-9075 result) |
| IQ-9075 whole policy call, resident on the NPU | IQ-9075 | `iq9/work/optionB/resident/resident_gframe_loop.py` | -- | scripts only (no published IQ-9075 result) |
| IQ-9075 action-head fix (bake the embodiment id) | IQ-9075 | `iq9/work/optionB/ah_fix/` | -- | scripts only (no published IQ-9075 result) |
| "What Hz means": control rate vs decision rate vs replan cadence | -- | no script: RoboCasa `control_freq=20`, GR00T's 16-step action horizon, and the whole-call times above | README, RESULTS.md | DERIVED |
| "Engine benchmark vs whole call": rate lost | all | arithmetic on the two rows above (`rate_lost_vs_engine_sum_pct` in each e2e record) | e2e records | regenerates |

## Accuracy

| Measurement | Boards | Script | Result | Status |
|---|---|---|---|---|
| Closed loop, 20 seeds, matched scenes + noise: 5090 bf16 reference, Orin fp16, Thor fp16 / fp8 / bf16 control (+ a 2026-10-12 bf16 re-run) | 5090, Orin, Thor | `accuracy/run_closed_loop.py`, `accuracy/finalize.py`, `accuracy/make_summary.py` | `results/accuracy/` (+ every action chunk in `results/accuracy/runs/`) | regenerates |
| Closed loop, IQ-9075 fp16 | IQ-9075 | `iq9/work/closed_loop_m4/m4_run.py --backend iq9` (fine-tuned bins built in `iq9/work/closed_loop_m4/build/`) | -- | scripts only (no published IQ-9075 result) |
| int8 / int16 closed loop (quantized action head) | IQ-9075 | `iq9/work/closed_loop_m7/m7_run.py` | -- | scripts only (no published IQ-9075 result) |
| int8 / int16 per-graph cosine vs fp16 | IQ-9075 | `iq9/work/closed_loop_m7/build_*.sh`, `gate_*.py` | -- | scripts only (no published IQ-9075 result) |
| int8 / int16 views (per-joint error, one trajectory) | IQ-9075 | `iq9/int8_int16/quant_views.py` | -- | scripts only (no published IQ-9075 result) |
| SmoothQuant on the LLM (int8 / int16 activations) | IQ-9075 | `iq9/work/closed_loop_m8/` | -- | scripts only (no published IQ-9075 result) |
| Denoise deviation vs a 64-step reference (solver error, not task success) | 5090 | the denoise sweep | sweep records | regenerates (full TensorRT) |

## Profiling

| Measurement | Boards | Script | Result | Status |
|---|---|---|---|---|
| Matmul vs glue, unfused (per-op cycles) | IQ-9075 | `iq9/profile/qnn_op_share.py` | -- | scripts only (no published IQ-9075 result) |
| Matmul vs glue, as TensorRT reports it (fused) | 5090, Thor, Orin | `profiling/compute_vs_glue/` | `results/profiling/groot_stage_compute_vs_glue_2026-10-09.json` | published record |
| Integer vs FP instructions (ncu SASS), incl. W4A16 LLM | Thor, Orin | `profiling/intfp_mix/` | `results/profiling/groot_intfp_instruction_mix_2026-10-09.json` | published record |
| Hexagon int vs FP, hand-coded HMX GEMM + HVX FAST loops | IQ-9075 | `iq9/hexagon_microbench/` | -- | scripts only (no published IQ-9075 result) |
| DDR bandwidth per stage | Thor, Orin | `profiling/ddr_bandwidth/` | `results/profiling/groot_ddr_bandwidth_2026-09-24.json` | regenerates (GB/s from the recorded actmon + calibration) |
| Thor DiT anomaly: roofline + kernel scan | Thor (roofs on all GPUs) | `profiling/thor_dit_anomaly/`, `profiling/compute_vs_glue/roofline_prof.py` | `results/profiling/thor_dit_anomaly_CLOSED_2026-08-27.json` | published record |

## Not included

- **Accelerators other than RTX 5090, Jetson AGX Thor, Jetson AGX Orin and Qualcomm IQ-9075** are not part of this repo.
- **Qualcomm SDK code**: the QAIRT/QNN SDK, its SampleApp utilities (needed to build `resident_orch`) and the Hexagon
  SDK example skeleton are licensed by Qualcomm; get them from Qualcomm and follow `iq9/README.md`.
- **Model weights, ONNX, engines, context binaries**: downloaded or rebuilt by the scripts; see [docs/SETUP.md](SETUP.md).
