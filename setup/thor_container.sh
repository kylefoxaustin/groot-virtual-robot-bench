#!/usr/bin/env bash
# Jetson AGX Thor (JetPack 7): run a command inside the NGC PyTorch container the published Thor speed
# numbers used (TensorRT 10.13, torch 2.8 nv25.08), with $GVRB_ROOT and $GVRB_WORK mounted at the same paths.
# Only transformers, pydantic, numpydantic and numpy are pinned below; the published Thor run did not record the
# versions of the other pip packages it installed into the container, so yours may differ (they are not on the timed path).
#   bash setup/thor_container.sh 'bash speed/run_board.sh thor'
set -euo pipefail
cd "$(dirname "$0")"; source ./versions.env; export GROOT_PY=python; source ./env.sh
docker run --rm --gpus all --ipc=host -v "$GVRB_ROOT:$GVRB_ROOT" -v "$GVRB_WORK:$GVRB_WORK" -w "$GVRB_ROOT" \
  -e GVRB_ROOT -e GVRB_WORK -e GROOT_REPO -e GROOT_MODEL -e GROOT_FT_MODEL -e GROOT_ONNX -e GROOT_STEP_NPZ \
  -e GROOT_SHIM -e JETSON=1 -e GROOT_PY=python -e PYTHONPATH="$GROOT_SHIM:$GROOT_REPO" \
  "$THOR_CONTAINER" bash -c "set -euo pipefail
pip install -q peft transformers==4.51.3 pydantic==2.13.4 numpydantic==1.10.0 diffusers timm sentencepiece protobuf av albumentations einops
pip install -q numpy==1.26.4
$*"
