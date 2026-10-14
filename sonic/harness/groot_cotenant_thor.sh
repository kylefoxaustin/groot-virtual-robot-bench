#!/bin/bash
# GR00T N1.5 full-TensorRT co-tenant on the Jetson AGX Thor, inside the same NGC container as the Thor speed runs
# (setup/thor_container.sh), engines from bash speed/run_board.sh thor (build A).   usage: groot_cotenant_thor.sh OUT_JSON DURATION_S
# OUT_JSON must be under $GVRB_WORK or $GVRB_ROOT (the only paths the container mounts).
set -euo pipefail
G=$(cd "$(dirname "$0")/../.." && pwd); source "$G/setup/env.sh"
bash "$G/setup/thor_container.sh" "python sonic/harness/groot_cotenant.py $GVRB_WORK/speed/thor/engA $1 $2"
