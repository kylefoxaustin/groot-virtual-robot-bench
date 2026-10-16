#!/usr/bin/env bash
# Drone benchmark setup on one machine, after the GR00T setup (docs/SETUP.md). Idempotent; re-run after a failure.
#   1. pip-installs the drone extras (MuJoCo, PyOpenGL) into $GROOT_PY's environment
#   2. checks the shipped assets (drone/assets/MANIFEST.json)
#   3. exports the fp16 ONNX graphs and the GR00T input step if they are missing (x86 only; on a Jetson copy them over)
#   4. builds the three TensorRT engines the drone uses: ViT + LLM (296 tokens) and vlln + vl_self_attention (289 tokens)
#      into $DRONE_ENG (a few minutes; Orin ~15 min)
#   bash drone/setup.sh                                               # x86 + NVIDIA GPU, Orin (GROOT_PY=python3)
#   bash setup/thor_container.sh 'bash drone/setup.sh'                # Jetson AGX Thor (inside NVIDIA's container)
# env: SKIP_PIP=1 (packages already there), SKIP_ENGINES=1
set -euo pipefail
cd "$(dirname "$0")/.."; source setup/env.sh; source drone/env.sh
PY="${GROOT_PY:-python3}"
if [ "${SKIP_PIP:-0}" != 1 ]; then
  if [ "$(uname -m)" = aarch64 ] && [ -z "${VIRTUAL_ENV:-}" ] && [ ! -f /.dockerenv ]; then USER_FLAG=--user; else USER_FLAG=; fi
  "$PY" -m pip install -q $USER_FLAG -r drone/requirements-drone.txt
fi
"$PY" - <<'PY'
import hashlib, json, os
d = os.environ["DRONE_ASSETS"]; m = json.load(open(os.path.join(d, "MANIFEST.json")))
for f, v in m["files"].items():
    h = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
    print(("OK  " if h == v["sha256"] else "BAD ") + f)
    assert h == v["sha256"], f"{f}: sha256 mismatch -- re-clone or git checkout drone/assets"
PY
need_onnx=0
for f in eagle2/vit_fp16.onnx eagle2/llm_fp16.onnx action_head/vlln_vl_self_attention.onnx; do [ -f "$GROOT_ONNX/$f" ] || need_onnx=1; done
if [ $need_onnx = 1 ]; then
  if [ "$(uname -m)" = aarch64 ]; then
    echo "missing ONNX in $GROOT_ONNX. The export needs an x86 machine with an NVIDIA GPU (it reads NVIDIA's demo video," >&2
    echo "which has no aarch64 decoder here). Export once there and copy both files over:" >&2
    echo "  x86:  bash speed/export_onnx.sh && \$GROOT_PY speed/make_step0.py \$GROOT_STEP_NPZ" >&2
    echo "        rsync -a \$GROOT_ONNX \$GROOT_STEP_NPZ <jetson>:gvrb/     (into the Jetson's \$GVRB_WORK)" >&2
    exit 1
  fi
  bash speed/export_onnx.sh
fi
if [ ! -f "$GROOT_STEP_NPZ" ]; then
  if [ "$(uname -m)" = aarch64 ]; then
    echo "missing $GROOT_STEP_NPZ: make it on the x86 host (\$GROOT_PY speed/make_step0.py \$GROOT_STEP_NPZ) and copy it here" >&2; exit 1
  fi
  "$PY" speed/make_step0.py "$GROOT_STEP_NPZ"
fi
if [ "${SKIP_ENGINES:-0}" != 1 ]; then
  [ -f "$DRONE_ENG/llm_fp16.engine" ] || ONLY=vit_fp16,llm_fp16 "$PY" speed/build_engines.py "$DRONE_ENG" 296
  [ -f "$DRONE_ENG/vlln289/vlln_vl_self_attention.engine" ] || ONLY=vlln_vl_self_attention "$PY" speed/build_engines.py "$DRONE_ENG/vlln289" 289
fi
ls -la "$DRONE_ENG" "$DRONE_ENG/vlln289"
echo "drone setup done. Next: bash drone/run_fullcost.sh <label>   (drone/README.md)"
