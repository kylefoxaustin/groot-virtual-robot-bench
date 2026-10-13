# Speed benchmarks

**What you get:** how long one whole GR00T policy call takes on your GPU (camera image + instruction in, 16-step action
chunk out), at three compilation levels: plain PyTorch (*eager*), the three big modules in TensorRT (*hybrid*), and
NVIDIA's own all-TensorRT deployment path (*full TensorRT*). Each run builds the TensorRT engines twice, because
build-to-build variation is a few percent.

**Before you start:** [docs/SETUP.md](SETUP.md) done, and `source setup/env.sh` in this shell.

## On an x86 machine with one NVIDIA GPU

```bash
source setup/env.sh
# 1) once: export the 7 fp16 ONNX graphs and the single benchmark input
bash speed/export_onnx.sh
$GROOT_PY speed/make_step0.py $GROOT_STEP_NPZ     # prints a content hash; must equal results/inputs.json
# 2) two engine builds -> eager / hybrid / full-TensorRT, same process -> one record
bash speed/run_board.sh mygpu "clocks not locked"
```

The first argument is a free-text label used in file names (we used `5090`, `thor`, `orin`); the second is a note about
clocks that goes into the record. Your record lands in `$GVRB_WORK/speed/<label>/<label>_fulltrt_e2e.json`; compare it
with ours in [`results/speed/groot_5090_fulltrt_e2e_2026-10-12.json`](../results/speed/groot_5090_fulltrt_e2e_2026-10-12.json).

Lock clocks if you can (`sudo nvidia-smi -lgc/-lmc`) and say so in the note; the harness alternates modes in one process
so drift cancels either way.

`run_board.sh` stops with a clear message if the ONNX export or the input file is missing. (Our published 5090 run read
the same frame straight from the demo dataset; `ALLOW_DATASET_INPUT=1` restores that, x86 only.)

## On a Jetson

Export the ONNX and the input once on an x86 host (step 1 above) and copy them over, so every board builds from
byte-identical graphs:

```bash
rsync -a $GROOT_ONNX $GROOT_STEP_NPZ <jetson>:gvrb/    # into the Jetson's $GVRB_WORK (default ~/gvrb)
GROOT_PY=python3 bash speed/run_board.sh orin "jetson_clocks MAXN"                 # Orin
bash setup/thor_container.sh 'bash speed/run_board.sh thor "jetson_clocks MAXN"'   # Thor
```

## More speed experiments

- **Denoise-steps sweep:** `bash speed/run_denoise_sweep.sh "clock note"` -- full-TensorRT sweep over 1-32 steps on
  both engine builds, the matching stage split, and the eager sweep. Needs the step-1 export. It reuses the engines in
  `$GVRB_WORK/speed/5090/` (building them if absent), so run `run_board.sh 5090 ...` first if you want to reuse those.
- **2 arms vs 4 arms:** `$GROOT_PY speed/arms_2v4/widen_action_head_onnx.py $GROOT_ONNX` makes a synthetic 4-arm action
  head (it reproduces our 4-arm ONNX weights exactly), then `$GROOT_PY speed/arms_2v4/bench_2v4.py <label>`
  (or `bash speed/arms_2v4/bench_2v4_trtexec.sh` where only `trtexec` is available).
- **Arms vs cameras:** `$GROOT_PY speed/multiarm/camera_token_scaling.py out.json` times every action-head module alone,
  the ViT at 1/2/4 camera views and the LLM at 256/512/1024 vision tokens.

Optional env vars: `ONLY=vit,llm` makes `speed/build_engines.py` build just those engines; `TENANCY_NOTE="..."` adds a
note about what else ran on the machine to the sweep records.

What the published numbers are and how they were measured: [RESULTS.md](../RESULTS.md) §1-4 and §7.
