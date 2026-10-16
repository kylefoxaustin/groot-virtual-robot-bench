#!/usr/bin/env bash
# x86 only: a small venv for the contention stand-ins (engine builds + workers) with TensorRT 10.16.1, the version our
# RTX 5090 stand-in runs used. TensorRT 11 (the GR00T venv) dropped the FP16 builder flag these fp32-ONNX stand-ins rely on.
# Jetsons need nothing: JetPack's python3 already has TensorRT + cuda-python.
#   bash drone/contention/setup_workers_x86.sh      -> $DRONE_WORK/venv-workers
set -euo pipefail
cd "$(dirname "$0")/../.."; source setup/env.sh
V="$DRONE_WORK/venv-workers"
PY="$(readlink -f "$(command -v python3.12 || command -v python3)")"
[ -x "$V/bin/python" ] || "$PY" -m venv "$V"
"$V/bin/pip" install -q --upgrade pip
"$V/bin/pip" install -q "tensorrt-cu13==10.16.1.11" "cuda-bindings==13.2.0" "numpy<2.6"
"$V/bin/python" -c "import tensorrt, numpy; from cuda.bindings import runtime; print('workers venv ok: tensorrt', tensorrt.__version__)"
