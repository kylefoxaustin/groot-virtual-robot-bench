#!/bin/bash
# GR00T N1.5 full-TensorRT co-tenant on the Jetson AGX Orin, native (as the Orin speed runs), engines from
# bash speed/run_board.sh orin (build A).   usage: groot_cotenant_orin.sh OUT_JSON DURATION_S
set -euo pipefail
G=$(cd "$(dirname "$0")/../.." && pwd); source "$G/setup/env.sh"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu JETSON=1
exec "$GROOT_PY" "$G/sonic/harness/groot_cotenant.py" "$GVRB_WORK/speed/orin/engA" "$1" "$2"
