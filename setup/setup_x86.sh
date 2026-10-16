#!/usr/bin/env bash
# x86 host with the RTX 5090: two venvs (policy/TensorRT, simulator), pinned upstreams, models, sim assets.
# Needs: NVIDIA driver new enough for CUDA 12.8 *and* the CUDA-13 TensorRT 11 wheels, python3.12 + python3.10,
# git, ~40 GB free in $GVRB_WORK (models ~13 GB, RoboCasa assets ~9 GB, ONNX/engines ~10 GB).
#   bash setup/setup_x86.sh                  everything (speed, closed loop, drone)
#   bash setup/setup_x86.sh --policy-only    the GR00T env + base model only (~20 GB): enough for the speed runs and drone/
set -euo pipefail
POLICY_ONLY=0; [ "${1:-}" = --policy-only ] && POLICY_ONLY=1
cd "$(dirname "$0")"; source ./versions.env; source ./env.sh
PYS="python3.12 python3.10"; [ $POLICY_ONLY = 1 ] && PYS=python3.12
for py in $PYS; do
  if ! command -v "$py" >/dev/null || ! "$py" -c "import venv, ensurepip" 2>/dev/null; then
    echo "ERROR: need $py with venv support (GR00T runs on 3.12, the RoboCasa simulator on 3.10)." >&2
    echo "       Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12-venv python3.10-venv" >&2
    exit 1
  fi
done
if [ $POLICY_ONLY = 1 ]; then bash ./fetch_upstream.sh; else bash ./fetch_upstream.sh --sim; fi
# venvs are made with the resolved interpreter: a symlinked python (e.g. uv's ~/.local/bin/python3.12) makes a broken venv
"$(readlink -f "$(command -v python3.12)")" -m venv "$GVRB_WORK/venv-groot"
"$GVRB_WORK/venv-groot/bin/pip" install -q --upgrade pip setuptools wheel
"$GVRB_WORK/venv-groot/bin/pip" install -r requirements-x86-groot.txt
# flash-attn has no wheel on PyPI. Install its prebuilt GitHub-release wheel for torch 2.9 + CUDA 12 + cp312 (no CUDA
# toolkit needed); only if that fails, build from source (needs nvcc from the CUDA 12.8 toolkit, and a long compile).
FA_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
"$GVRB_WORK/venv-groot/bin/pip" install "$FA_WHL" \
  || "$GVRB_WORK/venv-groot/bin/pip" install --no-build-isolation flash-attn==2.8.3
if [ $POLICY_ONLY = 1 ]; then
  PATH="$GVRB_WORK/venv-groot/bin:$PATH" bash ./fetch_models.sh base
  echo "x86 policy-only setup done. Next: docs/SPEED.md, or drone/README.md (bash drone/setup.sh)."; exit 0
fi
"$(readlink -f "$(command -v python3.10)")" -m venv "$GVRB_WORK/venv-robocasa"
"$GVRB_WORK/venv-robocasa/bin/pip" install -q --upgrade pip setuptools wheel
"$GVRB_WORK/venv-robocasa/bin/pip" install -r requirements-x86-robocasa.txt
"$GVRB_WORK/venv-robocasa/bin/pip" install -e "$GVRB_WORK/robosuite" -e "$GVRB_WORK/robocasa-gr1-tabletop-tasks"
(cd "$GVRB_WORK/robocasa-gr1-tabletop-tasks" && "$GVRB_WORK/venv-robocasa/bin/python" robocasa/scripts/download_tabletop_assets.py -y)
PATH="$GVRB_WORK/venv-groot/bin:$PATH" bash ./fetch_models.sh all
echo "x86 setup done. Next: docs/SPEED.md (export ONNX, then speed/run_board.sh) or docs/ACCURACY.md (closed loop)."
