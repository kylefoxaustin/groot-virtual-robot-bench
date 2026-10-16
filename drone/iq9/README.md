# The GR00T drone policy on the Qualcomm IQ-9075

The scripts to run the drone policy ([drone/](../)) on your own IQ-9075: the output gate, the full-cost decision rate,
closed-loop flights with the board making every decision, and contention with stand-in AI jobs on the second NSP.
**No IQ-9075 results are published in this repo.**

This is harder than the NVIDIA path. It builds on the GR00T port in [iq9/](../../iq9/README.md), it needs GR00T's Python
transforms on the board, and a few steps are run by hand on the board. Read each script's header before running it.

## What runs where

| Stage | Where | How |
|---|---|---|
| GR00T's input transforms (resize, Eagle processor, tokenizer) | board CPU (A78), torch 1 thread | `policy/drone_bframe_board.py` |
| SigLIP ViT, Qwen LLM (296-token build, right-padded) | one NSP, fp16 | context bins from the GR00T port |
| mlp1 GEMM + embedding / image-token scatter | board CPU, inside the orchestrator | `BFRAME` (`port/resident_orch_drone.patch`) |
| vlln + vl_self_attention at 289 tokens, mean-pool | same NSP, fp16; pool on the CPU | `port/build_vlln289.sh` + `BFRAME` |
| drone head (numpy float64) | board CPU, single-threaded BLAS | `drone/drone_head_np.py` |

One `BFRAME` call of the resident orchestrator runs ViT -> glue -> LLM -> vlln -> pool on one NSP. vl_self_attention has no
attention mask, so it is compiled at exactly 289 tokens and fed only the valid LLM rows; the causal LLM keeps its 296 build.

## What you need

| Thing | Version we used | Where it comes from |
|---|---|---|
| Qualcomm IQ-9075 EVK (two Hexagon v73 NSPs), Yocto with fastrpc, root ssh with key auth | -- | Qualcomm / Thundercomm |
| QAIRT / QNN SDK (x86 converter + board runtime) | **2.47.0.260601** | Qualcomm (account + license; not redistributed here). The Hexagon SDK is not needed |
| A working GR00T port on the board | -- | [iq9/README.md](../../iq9/README.md) sections 1-2: `ctx_fix/groot_vit_fp16.bin`, `groot_llm_fp16.bin`, `ctx_aaf0/groot_vit_aaf0_fp16.bin` (`build_groot_aaf0.sh`), `glue_constants/`, `io/position_ids.raw` |
| Board Python | 3.12 | numpy, torch (CPU; its `libopenblas.so.0` serves the orchestrator), opencv, and what `from gr00t.experiment.data_config import load_data_config` asks for (versions as in `setup/requirements-x86-groot.txt` where an aarch64 wheel exists; `shims/jetson` stubs flash_attn, decord, pytorch3d) |
| x86 host with an NVIDIA GPU and the drone setup | -- | [docs/SETUP.md](../../docs/SETUP.md) and `drone/setup.sh` (for the gate capture and the simulator) |
| `aarch64-linux-gnu-g++-11` | 11 | your distro |

```bash
source setup/env.sh && source drone/env.sh
export IQ9_HOST=<board>      # plus QAIRT_SDK_ROOT, IQ9_WORK, IQ9_QNN_ROOT, IQ9_PY_ROOT as for iq9/
source drone/iq9/env.sh      # DRONE_IQ9_WORK (host), DRONE_IQ9_BOARD_DIR (board, default /opt/gvrb-drone), DRONE_IQ9_*
```

On the board, `stage_board.sh` writes `$DRONE_IQ9_BOARD_DIR/env.sh`: `cd $DRONE_IQ9_BOARD_DIR && . ./env.sh` in every shell.
**Never SIGKILL a process that holds an NSP** (`resident_orch_drone*`, `qnn-net-run`): a killed DSP session can leave the
NPU degraded (slower, plausible-looking numbers) until reboot. Every script here stops the board side with `QUIT`.

## Steps

**1. Build (x86 host).**
`bash drone/iq9/port/build_vlln289.sh` (vlln + vl_self_attention at 289 tokens, arms `fix` and `aaf0`).
`bash drone/iq9/port/build_resident_orch.sh` (copies the SampleApp helpers from your SDK, applies the patches to
`iq9/work/optionB/src/resident_orch.cpp` in the work dir, cross-compiles `resident_orch_drone` and `resident_orch_drone_g`).

**2. Capture the gate inputs (GPU host).** `python3 drone/iq9/port/capture_gate_inputs.py` writes the 256 gate frames as
board raws: the exact Eagle tensors `drone/policy.py` feeds its backbone (bf16 pixels, ids right-padded to 296 with pad id
151643, mask 0).

**3. Stand-ins (contention only, x86 host).** `python3 drone/contention/make_standins.py`, then
`python3 drone/iq9/contention/make_dilfree.py` and `python3 drone/iq9/contention/make_seg1024_check.py` (both assert
fp32 equivalence with the canonical model), then `bash drone/iq9/contention/build_standins_qnn.sh`.

**4. Stage.** `bash drone/iq9/stage_board.sh` copies everything to `$IQ9_HOST:$DRONE_IQ9_BOARD_DIR`.

**5. Gate (board, then host).**
`$DRONE_IQ9_PY drone/iq9/policy/drone_bframe_board.py gate --out runs/gate_fix [--arm aaf0]`. Copy the dir back and run
`python3 drone/score_gate.py <dir>`: the same action-space rule as the GPUs, against the RTX 5090 bf16 reference.

**6. Full cost (board, then host).** `sh drone/iq9/policy/run_fullcost_iq9.sh` (as root: it sets the CPU governor). Arms
`fix` / `aaf0` on NSP0, `fix` on NSP1, and two streams on both NSPs at once. On the host:
`python3 drone/iq9/policy/score_fullcost_iq9.py <out>/s_fix_d0_r*` (and `--dual <d0> <d1> ...`). Each decision is timed
from the raw frame + prompt + state to the setpoints, transforms included; the scorer also checks every decision's output.

**7. Closed loop.** On the board, start the server detached:
`mkdir -p runs && setsid nohup $DRONE_IQ9_PY drone/iq9/policy/drone_server_iq9.py --out runs/server --device 0 > runs/server.out 2>&1 &`.
On the simulator host, open your own tunnel (`ssh -N -L 47124:127.0.0.1:47124 $IQ9_HOST`), then

```bash
$GROOT_PY drone/closed_loop.py --out $DRONE_RUNS/iq9_cl --arm iq9_K4 --policy 127.0.0.1:47124 --K 4        # lockstep
$GROOT_PY drone/closed_loop.py --out $DRONE_RUNS/iq9_cl --arm iq9_K15 --policy 127.0.0.1:47124 --K 15
$GROOT_PY drone/closed_loop.py --out $DRONE_RUNS/iq9_cl --arm iq9_own --policy 127.0.0.1:47124 --d own [--dmode aligned]
python3 drone/iq9/policy/server_quit.py 127.0.0.1:47124
```

`--d own` uses each decision's board-side time (`board_wall`, network excluded) as its delay. Compare episodes with
`python3 drone/compare_closed_loop.py <yours>.jsonl results/drone/raw/closed_loop/<published arm>.jsonl`.

**8. Contention (board, then host).** The nav runs back to back on NSP0; stand-ins run on NSP1.
- Check each compiled stand-in first: `npu_worker.py check ... --keep <dir>`, then on the host
  `python3 drone/iq9/contention/score_check.py <dir> <worker.json> seg512|seg1024|radar`.
- Sessions: `sh drone/iq9/contention/run_session.sh TAG drone/iq9/contention/plans/scan.txt` (as root). `plans/` has
  `scan` and `all` (the GPU schedules: score with `drone/contention/score_run.py <run dir>`),
  `radar_shares_scan_nsp` and `attribution` (score with `bin_session.py <run dir> <plan>.windows.json`).
- `contention/profile_exec.sh` splits one stand-in call into list read / input populate / execute / output write, with
  `perf`, and checks that `EXECF` writes the same bytes as `EXEC`.

## Traps worth knowing

- **BLAS thread spin.** A multi-threaded numpy OpenBLAS in a host helper next to an NPU loop leaves workers spinning after each
  call; they steal the cores the orchestrator's mlp1 GEMM needs. The scripts set `OPENBLAS_NUM_THREADS=1` before numpy loads.
- **Per-byte output write.** The SampleApp helper behind `EXEC` writes outputs element by element. For a large output (the
  seg stand-in) that host work can become a co-tenant of its own. `EXECF` writes with one `fwrite`; `EXECNOP` / `EXECONLY` separate host
  work from NPU work (`plans/attribution.txt`).
- **Dilated convolutions.** The seg stand-in's dilated convs (dilation rates comparable to the feature-map size) can be very slow on v73 HTP. `make_dilfree.py` rewrites them exactly; the check scores against the original model's references.
- **Pin the nav's host threads** (`NAV_CPUS=...` for `run_session.sh`, `runpin` in a plan) so a co-tenant's host work
  does not land on them; record what you pinned.
- Stop with `QUIT` (`server_quit.py`, stop files); SIGTERM only on a hang.
- `graph_names` in an HTP config must equal the graph name, or every option is silently ignored; the build scripts check
  that the `fix` and `aaf0` bins differ. Never convert with `--float_bitwidth 16`.
- `--device N` pins the NSP strictly: an unknown id is an error, never a fallback.
- The full-cost path feeds fp32 pixels from the board's transforms; the gate feeds the GPU path's bf16 pixels. Both are
  scored against the same reference.
