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

## GEAR-SONIC whole-body controller (Unitree G1, MuJoCo)

| Measurement | Boards | Script | Result | Status |
|---|---|---|---|---|
| 50 Hz closed loop, unloaded: step time, compute / slot misses, late starts, max gap, falls (rule window + band drop), tracking | 5090, Thor, Orin | `sonic/harness/run_closed_loop_{x86,jetson}.py`, `sonic/scoring/`, `sonic/make_sonic_table.py` | `results/sonic/sonic_same_span_table_2026-10-14.json` (+ per-tick logs in `results/sonic/raw/`) | regenerates (timing and fall columns from the per-tick logs and per-run simulator records; per-phase joint errors are published per run as measured) |
| Same, CPU saturated and GPU co-tenant (GR00T N1.5 full TensorRT) | Thor, Orin | `sonic/harness/jetson_runs.sh`, `sonic/harness/groot_cotenant.py`, `sonic/scoring/score_groot.py` | same table + `results/sonic/sonic_{thor,orin}_2026-10-14.json` (GR00T rates) | regenerates (table); published record (GR00T rates) |
| Busy-spin diagnostic, CPU saturated with the deploy's FIFO request disabled (n = 1 per board) | Thor, Orin | `sonic/harness/jetson_runs.sh diag` | same table | regenerates |
| Engine correctness vs the 5090 tensors (encoder, decoder, planner; TF32 diagnosis) | Thor, Orin | `sonic/gate/` | `results/sonic/sonic_{thor,orin}_2026-10-14.json` (`A2_correctness`) | published record (the reference sets are not published) |
| Band-drop falls: Thor 3/10, Orin 1/10 vs 5090 0/3 (Fisher p) | all | `sonic/make_sonic_table.py` | same table | DERIVED (answered by the controlled test below) |
| First campaign re-scored under the D1 fall rule (any fall-reset from band-off to stop) | 5090, Thor, Orin | `sonic/make_sonic_table.py` | same table (`falls_d1*` per group) | regenerates |
| Controlled band-drop test: release at D = 10-100 ms × 3 runs, falls per D, Clopper-Pearson, Fisher, bit-identical runs, body-state check | 5090, Thor, Orin | `sonic/harness/sim_driver.py` (`SIM_RELEASE_DELAY_MS`), `sonic/harness/banddrop_x86.sh`, `sonic/harness/jetson_runs.sh banddrop` | same table (`band_drop_controlled_DERIVED`) + `results/sonic/sonic_banddrop_controlled_2026-10-14.json` | regenerates (counts from the per-run simulator records; intervals and tests DERIVED) |
| Late-start fix (control thread SCHED_FIFO 90 + sleeping main loop): unloaded, CPU saturated, GPU co-tenant vs stock | Thor, Orin | `sonic/harness/latefix.patch`, `sonic/harness/jetson_runs.sh fix` | same table (`<board>/fix_*`, `<board>/stock_sameday_unloaded`) + `results/sonic/sonic_jetson_latefix_2026-10-14.json` | regenerates (table); published record (/proc scheduling readbacks, GR00T rates) |

## GR00T drone (frozen GR00T N1.5 backbone + trained head, MuJoCo quadrotor)

| Measurement | Boards | Script | Result | Status |
|---|---|---|---|---|
| Full-cost decision rate (raw frame + prompt + state to setpoints, host included), 3 × 64 decisions | 5090, Thor, Orin | `drone/run_fullcost.sh`, `drone/drone_fulltrt.py fullcost` | `results/drone/drone_table_2026-10-15.json` (`full_cost`) + `results/drone/raw/fullcost/` | regenerates (host + glue share and engine sums DERIVED) |
| Same on the RTX 5090 pinned to P-cores, plus 75 s loops pinned and unpinned | 5090 | `drone/drone_fulltrt.py fullcost` / `loop` under `taskset` | same table + `results/drone/raw/{fullcost,loops}/` | regenerates |
| Output gate in action space vs the RTX 5090 eager bf16 reference (RMS, per-frame p95, zero-image null, planted fault) | 5090, Thor, Orin | `drone/drone_fulltrt.py gate`, `drone/score_gate.py` | same table (`output_gate`) + `results/drone/raw/gate/` | regenerates (from the published setpoints; the features themselves are not published) |
| Closed-loop reach, 40 held-out episodes: lockstep K = 4/15, K sweep, TensorRT vs eager, NVIDIA-EGL reruns | 5090 | `drone/closed_loop.py` | same table (`closed_loop`) + `results/drone/raw/closed_loop/rtx5090_*.jsonl` | regenerates (reach MEASURED; agreement and McNemar DERIVED) |
| Closed-loop reach with the board serving every decision: lockstep K = 4 and back-to-back at the measured latency | Thor, Orin | `drone/drone_fulltrt.py serve` + `drone/closed_loop.py --policy` | same + `results/drone/raw/closed_loop/{thor,orin}_*.jsonl` | regenerates |
| Moving while deciding: injected delay d = 1-17 ticks, naive and aligned, vs no delay | 5090 | `drone/closed_loop.py --d N --dmode M` | same + `results/drone/raw/closed_loop/rtx5090_delay_*.jsonl` | regenerates |
| Nav rate with a 50 MP inspection burst, radar fusion (20 Hz, back to back), and all three jobs at once; gated per window | 5090, Thor, Orin | `drone/contention/` | same table (`contention`) + `results/drone/raw/contention/` | regenerates (rates from per-decision times; stand-ins regenerate from seed 20261015) |
| IQ-9075 drone port, gate, full cost, closed loop, second-NSP contention | IQ-9075 | `drone/iq9/` | -- | scripts only |

## Not included

- **Accelerators other than RTX 5090, Jetson AGX Thor, Jetson AGX Orin and Qualcomm IQ-9075** are not part of this repo.
- **Qualcomm SDK code**: the QAIRT/QNN SDK, its SampleApp utilities (needed to build `resident_orch`) and the Hexagon
  SDK example skeleton are licensed by Qualcomm; get them from Qualcomm and follow `iq9/README.md`.
- **Model weights, ONNX, engines, context binaries**: downloaded or rebuilt by the scripts; see [docs/SETUP.md](SETUP.md). The one
  exception is the drone's small trained head and gate set (`drone/assets/`), which cannot be regenerated bit for bit.
