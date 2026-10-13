# Accuracy benchmarks (closed loop)

**What you get:** GR00T drives a simulated Fourier GR1 humanoid through 20 seeded episodes of *pick up the milk, put it
in the microwave, close the door* (RoboCasa, 720 simulator steps each). You get, per seed, whether the task succeeded,
plus every action chunk the policy produced -- and a seed-by-seed comparison with our RTX 5090 reference run, which ships
in [`results/accuracy/runs/rtx5090_bf16_reference`](../results/accuracy/runs/rtx5090_bf16_reference).

**Why it's comparable across machines:** every backend gets the same diffusion noise (a shared noise bank) and the client
re-seeds every scene exactly, so two backends differ only in arithmetic. *Same outcome* = on how many of the 20 seeds a
backend succeeded or failed exactly as the reference did.

**Before you start:** [docs/SETUP.md](SETUP.md) done (x86 part), and `source setup/env.sh` in this shell. The simulator
pauses while the policy thinks, so slow hardware gives the same answers, just later.

## On one x86 machine with an NVIDIA GPU

```bash
source setup/env.sh
# 1) the shared noise bank (the sha256 it prints must match results/inputs.json)
$GROOT_PY accuracy/gen_noise.py --n_episodes 20 --seed_a 5000 --out_a $GVRB_WORK/noise_bank.npy --only_a
# 2) policy server and simulator both on this machine (--precision fp16, bf16 or fp8)
python3 accuracy/run_closed_loop.py --server local --precision bf16 \
    --noise_bank $GVRB_WORK/noise_bank.npy --out_dir $GVRB_WORK/runs/mygpu_bf16
# 3) compare with our shipped RTX 5090 reference
#    (the 5090 reference predates scene hashing, so scenes are checked against our Orin run instead)
$GROOT_PY accuracy/finalize.py --board_name "my GPU" --precision bf16 --dir_board $GVRB_WORK/runs/mygpu_bf16 \
    --dir_5090 results/accuracy/runs/rtx5090_bf16_reference --board_calllog $GVRB_WORK/runs/mygpu_bf16/work/server_calllog.json \
    --scene_ref results/accuracy/runs/orin_fp16 --out $GVRB_WORK/runs/finalize_mygpu.json
```

`--server ref` instead of `local` runs the exact reference server we used for the published 5090 bf16 run.

## With a Jetson as the policy server

The simulator stays on the x86 host; the Jetson runs the policy and talks to it over the LAN (ssh + TCP 5571).

```bash
python3 accuracy/run_closed_loop.py --server remote --precision fp16 --ssh orin --board_ip <orin-ip> \
    --remote_repo <this repo on orin> --remote_isaac <Isaac-GR00T on orin> --remote_model <ft checkpoint on orin> \
    --remote_python python3 --noise_bank $GVRB_WORK/noise_bank.npy --out_dir $GVRB_WORK/runs/orin_fp16
#  Thor: --precision fp16 (our headline run), fp8, or bf16; --remote_python <thor venv>/bin/python
$GROOT_PY accuracy/finalize.py --board_name "Jetson AGX Orin" --precision fp16 --dir_board $GVRB_WORK/runs/orin_fp16 \
    --dir_5090 results/accuracy/runs/rtx5090_bf16_reference --board_calllog $GVRB_WORK/runs/orin_fp16/work/server_calllog.json \
    --scene_ref results/accuracy/runs/thor_fp16 --out $GVRB_WORK/runs/finalize_orin.json
```

`--server remote` sets the board's `PYTHONPATH` to `shims/jetson`, runs ssh with `-n` (stdin detached) and stops the real
server process, not its wrapper shell. On Thor use the venv described in [docs/SETUP.md](SETUP.md).

## Good to know

- `--scene_ref` checks the first observation of every episode against another run (or an `accuracy/scene_probe.py`
  output), proving both runs saw identical scenes. In the shipped runs `server_calllog.json` sits at the top of each run
  directory; your own runs keep it in `work/`.
- The fp8 server converts every eligible `nn.Linear` to a genuine e4m3 `torch._scaled_mm` GEMM and refuses to start if
  fp8 cannot engage; `fp8_proof.json` counts the fp8 GEMMs.
- Stop runs with Ctrl-C or SIGTERM, never SIGKILL: a killed GPU process can leave a Jetson in a degraded state.
- `accuracy/make_summary.py` rebuilds our summary table from the `finalize_*.json` files.
- An IQ-9075 backend uses the same noise bank and client; see [iq9/README.md](../iq9/README.md).

What the published numbers are: [RESULTS.md](../RESULTS.md) §5.
