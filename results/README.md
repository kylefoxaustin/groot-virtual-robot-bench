# results/

Every published number lives here as JSON. [RESULTS.md](../RESULTS.md) is generated from these files, and
`tools/check_results.py` rebuilds the ones that have raw data (see [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)).

| Path | What it is |
|---|---|
| `inputs.json` | content hashes of the exact inputs: the speed benchmark's input frame, the closed-loop noise bank, the model revisions |
| `speed/groot_<board>_fulltrt_e2e_2026-10-12.json` | the headline speed records (eager / hybrid / full TensorRT, two engine builds) |
| `speed/n15_5090_*` | denoise-steps sweeps and stage splits on the RTX 5090 (`n15` = GR00T N1.5) |
| `speed/raw/` | the raw harness output the speed records are rebuilt from; `_A` / `_B` = the two independent TensorRT engine builds |
| `speed/onnx_manifest.json` | checksums of our ONNX export, to compare with yours |
| `accuracy/finalize_<run>_vs_<reference>.json` | one board run scored seed by seed against a reference run |
| `accuracy/closed_loop_summary_2026-09-18.json` | the summary table, rebuilt from the `finalize_*` files by `accuracy/make_summary.py` |
| `accuracy/runs/<run>/` | the raw closed-loop runs: per-seed success (`rollout_summary.json`), every action chunk (`episode_*_actions.npy`), the server's per-call timings (`server_calllog.json`) |
| `sonic/sonic_<board>_<date>.json` | GEAR-SONIC whole-body controller records (RTX 5090 reference, Jetson AGX Thor, Jetson AGX Orin): setup, versions, engine hashes, engine-correctness gate, loaded arms, predictions vs outcomes |
| `sonic/sonic_same_span_table_2026-10-14.json` | every SONIC run and group, rebuilt from `sonic/raw/` by `sonic/make_sonic_table.py` |
| `sonic/raw/<board>/<run>__*` | per-tick timing (`__ticks.csv.gz`), planner calls (`__planner.csv.gz`) and the per-run simulator record (`__run.json`); times relative to control start |
| `drone/drone_table_2026-10-15.json` | every drone number (full cost, output gate, closed loop, contention), rebuilt from `drone/raw/` by `drone/make_drone_table.py` |
| `drone/drone_setup_2026-10-15.json` | how the drone runs were set up: machines, software, clocks, engine checksums, protocols, the 30 Hz target, caveats |
| `drone/raw/` | per-decision timings (`fullcost/`, `loops/`), the gate setpoints (`gate/`), one row per flight (`closed_loop/`), per-decision times and co-tenant windows (`contention/`) |
| `drone/standins_MANIFEST.json` | sha256 of the seeded contention stand-ins, to compare with `drone/contention/make_standins.py` |
| `profiling/` | profiling records, published as measured. `_CLOSED_` in a file name means the investigation is finished |

Dates in file names are when the measurement was made. Most records carry a `tag` (MEASURED / DERIVED / SOURCED), plus software versions and method. Clocks are in the speed
records; the arms-vs-cameras and profiling records (RESULTS.md §4, §6, §7) do not record them.
