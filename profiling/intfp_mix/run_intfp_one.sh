#!/bin/bash
# Profile ONE engine's int/fp mix. Usage (as sudo): sudo bash run_intfp_one.sh <engine-basename>
# Engines are read from $GVRB_BOARD_WORK; parse_intfp.py is taken from this script's directory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${GVRB_BOARD_WORK:-$HOME/gvrb-board}"; [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ] && D="${GVRB_BOARD_WORK:-/home/$SUDO_USER/gvrb-board}"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
[ -n "$NCU" ] || { echo "ncu not found" >&2; exit 1; }
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
eng="$D/${1:?usage: run_intfp_one.sh <engine-basename>}"; nm=$(basename "$eng"); OUT="$D/intfp_one_$nm.txt"
[ -f "$eng" ] || { echo "missing engine $eng" >&2; exit 1; }
M="smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,smsp__sass_thread_inst_executed.sum,sm__inst_executed_pipe_tensor.sum"
echo "profiling $nm on $(hostname)"
rc=0
"$NCU" --metrics "$M" --csv --page raw --target-processes all "$TRTEXEC" --loadEngine="$eng" --iterations=1 --warmUp=0 --noDataTransfers > "$D/_ncu_$nm.csv" 2>"$D/_ncu_$nm.log" || rc=$?
if [ $rc -ne 0 ]; then echo "ERROR: ncu/trtexec exited $rc for $nm (log: $D/_ncu_$nm.log)" >&2; exit $rc; fi
python3 "$HERE/parse_intfp.py" "$D/_ncu_$nm.csv" "$nm" | tee "$OUT"; chmod 644 "$OUT" "$D/_ncu_$nm.csv" 2>/dev/null || true
echo "DONE -> $OUT"
