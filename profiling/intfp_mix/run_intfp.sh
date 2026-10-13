#!/bin/bash
# GR00T int-vs-fp instruction mix via ncu, for every *.engine in $GVRB_BOARD_WORK. Run under sudo.
# Writes $GVRB_BOARD_WORK/intfp_results.txt (world-readable); parse_intfp.py is taken from this script's directory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${GVRB_BOARD_WORK:-$HOME/gvrb-board}"; [ -d /root ] && [ "$(id -u)" = 0 ] && D="${GVRB_BOARD_WORK:-${SUDO_USER:+/home/$SUDO_USER/gvrb-board}}"; D="${D:-$HOME/gvrb-board}"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
[ -n "$NCU" ] || { echo "ncu not found" >&2; exit 1; }
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
OUT="$D/intfp_results.txt"; : > "$OUT"
M="smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,smsp__sass_thread_inst_executed.sum,sm__inst_executed_pipe_tensor.sum"
echo "host=$(hostname) ncu=$NCU dir=$D" | tee -a "$OUT"
FAILED=0
for eng in "$D"/*.engine; do
  [ -f "$eng" ] || continue; nm=$(basename "$eng")
  echo "=== $nm ===" | tee -a "$OUT"
  rc=0
  "$NCU" --metrics "$M" --csv --page raw --target-processes all \
     "$TRTEXEC" --loadEngine="$eng" --iterations=1 --warmUp=0 --noDataTransfers > "$D/_ncu_$nm.csv" 2>"$D/_ncu_$nm.log" || rc=$?
  if [ $rc -ne 0 ]; then echo "  ERROR: ncu rc=$rc (log: $D/_ncu_$nm.log)" | tee -a "$OUT"; FAILED=$((FAILED+1)); continue; fi
  python3 "$HERE/parse_intfp.py" "$D/_ncu_$nm.csv" "$nm" | tee -a "$OUT"
done
chmod 644 "$OUT" 2>/dev/null || true; echo "DONE -> $OUT"
if [ $FAILED -ne 0 ]; then echo "ERROR: $FAILED engine(s) failed to profile" >&2; exit 1; fi
