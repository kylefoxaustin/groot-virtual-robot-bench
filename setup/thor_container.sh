#!/usr/bin/env bash
# Jetson AGX Thor (JetPack 7): run a command inside the NGC PyTorch container the published Thor speed
# numbers used (TensorRT 10.13, torch 2.8 nv25.08), with $GVRB_ROOT and $GVRB_WORK mounted at the same paths.
# Only transformers, pydantic, numpydantic and numpy are pinned below; the published Thor run did not record the
# versions of the other pip packages it installed into the container, so yours may differ (they are not on the timed path).
#   bash setup/thor_container.sh 'bash speed/run_board.sh thor'
set -euo pipefail
cd "$(dirname "$0")"; source ./versions.env; export GROOT_PY=python; source ./env.sh; source ../drone/env.sh
# The container ships a real flash-attn, so it gets shims/thor_container (decord + pytorch3d stubs only), not shims/jetson.
# THOR_DOCKER_ARGS: extra docker run arguments, e.g. "--network host" for the drone policy server (drone/README.md)
docker info >/dev/null 2>&1 || { echo "docker is not usable by $(id -un): install docker + the NVIDIA container runtime and add yourself to the docker group" >&2; exit 1; }
if [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ]; then echo "run this as yourself, not with sudo (HOME, and so GVRB_WORK, would point at /root)" >&2; exit 1; fi
[ -d "$GVRB_WORK" ] || { echo "GVRB_WORK=$GVRB_WORK does not exist: run bash setup/fetch_upstream.sh first (docs/SETUP.md)" >&2; exit 1; }
docker run --rm --gpus all --ipc=host ${THOR_DOCKER_ARGS:-} -v "$GVRB_ROOT:$GVRB_ROOT" -v "$GVRB_WORK:$GVRB_WORK" -w "$GVRB_ROOT" \
  -e GVRB_ROOT -e GVRB_WORK -e GROOT_REPO -e GROOT_MODEL -e GROOT_FT_MODEL -e GROOT_ONNX -e GROOT_STEP_NPZ \
  -e GROOT_SHIM="$GVRB_ROOT/shims/thor_container" -e JETSON=1 -e GROOT_PY=python -e PYTHONPATH="$GVRB_ROOT/shims/thor_container:$GROOT_REPO" \
  -e DRONE_WORK -e DRONE_ENG -e DRONE_ASSETS -e DRONE_STANDINS -e DRONE_RUNS -e DRONE_TORCH_THREADS -e NO_ALBUMENTATIONS_UPDATE -e PYTHONDONTWRITEBYTECODE=1 \
  "$THOR_CONTAINER" bash -c "set -euo pipefail
pip install -q peft transformers==4.51.3 pydantic==2.13.4 numpydantic==1.10.0 diffusers timm sentencepiece protobuf av albumentations einops
pip install -q numpy==1.26.4
trap 'chown -R $(id -u):$(id -g) \"\$GVRB_WORK\" 2>/dev/null || true' EXIT   # outputs belong to you, not root
$*"
