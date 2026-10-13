# Profiling: where GR00T's time actually goes

The analysis behind *why* the numbers look the way they do: per-engine timing on every board, the Thor DiT anomaly, matmul vs glue, integer vs
floating-point instructions, and DDR bandwidth. Unlike `speed/` (clean harnesses), most of these are the scripts as they
ran, with private paths swapped for `setup/env.sh` variables (`GVRB_WORK` on the host, `GVRB_BOARD_WORK` on a Jetson).

| What | Board | Run | Output / record |
|---|---|---|---|
| Stage-input tensors for the standalone engine runs (step 0 of every row below that times ViT / LLM / DiT engines) | any CUDA host with the policy env | `$GROOT_PY compiled_fleet/dump_stage_inputs.py` (from `$GROOT_STEP_NPZ`), then `python3 compiled_fleet/mkref.py {vit,llm,dit}` (fp32 onnxruntime-CPU reference) | `$GVRB_WORK/trtwork/io/*.raw`, `$GVRB_WORK/trtwork/ref_*.npy` (see "Stage inputs" below) |
| Engines timed standalone (ViT, LLM, DiT step), 3-6 builds each | Thor | `compiled_fleet/thor/thor_groot.sh` (trtexec, strongly typed / fp16 / opt-5 variants), `thor_gate.py`, `thor_summarize.py`, `thor_final.py` | `results/profiling/groot_compiled_fleet_2026-08-27.json` |
| same | Orin | `compiled_fleet/orin/build_one.sh`, `gate.py`, `summarize.py`; e2e reconcile `bench_groot_reconcile.py` | same |
| same | RTX 5090 (Python TensorRT API, no trtexec) | `compiled_fleet/rtx5090/trt_groot.py`, `repeat_timing.py`, `xcheck_timing.py`, `chain_cost.py`, `dit4_check.py`, `summarize_5090.py`; cross-board table `compiled_fleet/crossboard.py` | same |
| Thor DiT anomaly (roofline + kernel scan) | Thor | `thor_dit_anomaly/dit_scan.py` (reads a trtexec `--exportProfile` dump named `dit_prof.json`), `inspect_dit*.py` (layer-info dumps from `thor_groot.sh`) | `results/profiling/thor_dit_anomaly_CLOSED_2026-08-27.json` |
| Matmul vs glue, NVIDIA (per-layer time) | Thor / Orin: trtexec `--dumpProfile --exportProfile`; 5090: TensorRT IProfiler | `compute_vs_glue/build_fp16_engine.py` (default-fusion fp16 build), `roofline_prof.py <engine> <stage>` (5090), `categ_trtexec_layers.py <profile.json>` / `categ_trtexec_layers_v2.py` (Jetson) | `results/profiling/groot_stage_compute_vs_glue_2026-10-09.json` |
| Matmul vs glue, IQ-9075 (per-op cycles) | IQ-9075 | `iq9/profile/qnn_op_share.py` | scripts only, no published result |
| Integer vs FP instructions (ncu SASS counters), incl. W4A16 LLM | Thor, Orin (root) | `intfp_mix/run_intfp.sh` (every `*.engine` in `$GVRB_BOARD_WORK`), `run_intfp_one.sh <engine>`, `parse_intfp.py` | `results/profiling/groot_intfp_instruction_mix_2026-10-09.json` |
| Hexagon int vs FP (static disassembly) | IQ-9075 | `iq9/hexagon_microbench/disasm_mix.py` | scripts only, no published result |
| DDR bandwidth per stage | Thor, Orin (root) | `ddr_bandwidth/actmon_stage.sh` per stage engine, `gpu_bw_cal.py` + `gpu_bw_multistream.py` (calibration), `derive_gbps.py` (GB/s) | `results/profiling/groot_ddr_bandwidth_2026-09-24.json` |

Notes
- `bench_groot_n15.py --power` and `bench_groot_reconcile.py --power` sample Jetson power rails through an optional
  external power sampler (not included): set `POWER_SAMPLER_DIR` to the directory of a module (`POWER_SAMPLER_MODULE`,
  default `rail_sampler`) exposing `RailSampler`. Power was not used for any published number in this repo.
- Scripts call their sibling scripts from their own directory (`thor/gw.sh`, `intfp_mix/run_intfp*.sh`), never from a
  work directory.
- ncu and actmon need root. jtop's EMC value reads 0 on JetPack 6/7; actmon is the working DDR counter.
- `derive_gbps.py` reproduces the published GB/s from the record's actmon + calibration data exactly
  (`tools/check_results.py` runs it). `actmon_stage.sh` re-implements the sampling wrapper from the record's documented
  method; the calibration and derivation scripts are the originals.

Stage inputs
- `thor/thor_groot.sh`, `thor/rep.sh`, `orin/build_one.sh` (trtexec `--loadInputs`), `rtx5090/trt_groot.py` and
  `mkref.py` read these files from `$GVRB_STAGE_IO` (default `$GVRB_WORK/trtwork/io`), raw little-endian in the engines'
  binding dtypes, and refuse to run if one is missing:

  | File | dtype | shape |
  |---|---|---|
  | `vit_pixel_values.raw` | float16 | 1x3x224x224 |
  | `vit_position_ids.raw` | int64 | 1x256 (`arange(256)`) |
  | `llm_inputs_embeds.raw` | float16 | 1x296x2048 |
  | `llm_attention_mask.raw` | int64 | 1x296 |
  | `dit_sa_embs.raw` | float16 | 1x49x1536 |
  | `dit_vl_embs.raw` | float16 | 1x296x2048 |
  | `dit_timesteps_tensor.raw` | int64 | 1 |

- `compiled_fleet/dump_stage_inputs.py` writes them by running one eager `policy.get_action()` on `$GROOT_STEP_NPZ` and
  capturing the tensors at the ViT / LLM / DiT module seams (DiT: first denoise step), cast as the speed harness casts
  them. The published runs used files of the same provenance (the same PickNPlace step) whose md5s are not shipped, so
  regenerated files are equivalent but not provably byte-identical. Timing of these fixed-shape engines does not depend
  on tensor content, and the gates compare against `ref_<stage>.npy` that `mkref.py` computes from the same files.
- The DDR-bandwidth and int/fp-mix rows replay already-built engines (`--loadEngine`) and need no input files.
