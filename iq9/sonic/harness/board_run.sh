#!/bin/sh
# board_run.sh <outdir> <sonic_bench args...>   (ON the board, from $IQ9_SONIC_BOARD_DIR; env.sh sourced)
# Refuses on a busy board; graceful timeout only (SIGTERM, never SIGKILL: a killed DSP session can leave the NSP degraded).
. $IQ9_SONIC_BOARD_DIR/harness/board_quiet.sh
OUT=$1; shift; mkdir -p "$OUT"
pgrep -x sonic_bench >/dev/null && { echo "REFUSING: sonic_bench already running"; exit 3; }
require_quiet_board || exit 3
export LD_LIBRARY_PATH=$IQ9_QNN_ROOT/lib:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=$IQ9_QNN_ROOT/skel
cat /proc/loadavg > "$OUT/loadavg_before.txt"
timeout -s TERM 600 $IQ9_SONIC_BOARD_DIR/bin/sonic_bench --backend $IQ9_QNN_ROOT/lib/libQnnHtp.so \
  --system_library $IQ9_QNN_ROOT/lib/libQnnSystem.so --out "$OUT" "$@" 2> "$OUT/stderr.txt"
RC=$?; echo "rc=$RC" > "$OUT/rc.txt"; tail -3 "$OUT/stderr.txt"; echo "rc=$RC"
cat /sys/class/remoteproc/*/state | tr '\n' ' '; echo
