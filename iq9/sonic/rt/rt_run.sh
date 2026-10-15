#!/bin/sh
# rt_run.sh <outdir> <timeout_s> <sonic_rt args...>   (ON the board, env.sh sourced)
# Refuses on a busy board unless ALLOW_LOAD=1 (the CPU-saturated arms start their own load first); graceful timeout only
# (SIGTERM, never SIGKILL). Records load average, remoteproc state and new kernel messages (by timestamp, not by line count:
# a saturated dmesg ring buffer keeps the same number of lines, so a line-count check sees nothing).
. $IQ9_SONIC_BOARD_DIR/harness/board_quiet.sh
OUT=$1; TO=$2; shift 2; mkdir -p "$OUT"
for p in sonic_rt sonic_bench qnn-net-run; do pgrep -x $p >/dev/null && { echo "REFUSING: $p already running"; exit 3; }; done
[ -n "$ALLOW_LOAD" ] || require_quiet_board || exit 3
export LD_LIBRARY_PATH=$IQ9_QNN_ROOT/lib:$IQ9_SONIC_BOARD_DIR/lib:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=$IQ9_QNN_ROOT/skel
cat /proc/loadavg > "$OUT/loadavg_before.txt"; T0=$(cut -d' ' -f1 /proc/uptime)
cat /sys/class/remoteproc/*/state | tr '\n' ' ' > "$OUT/rproc_before.txt"
timeout -s TERM "$TO" $IQ9_SONIC_BOARD_DIR/bin/sonic_rt --backend $IQ9_QNN_ROOT/lib/libQnnHtp.so \
  --system_library $IQ9_QNN_ROOT/lib/libQnnSystem.so --out "$OUT" "$@" 2> "$OUT/stderr.txt" > "$OUT/stdout.txt"
RC=$?; echo "rc=$RC" > "$OUT/rc.txt"
cat /proc/loadavg > "$OUT/loadavg_after.txt"
cat /sys/class/remoteproc/*/state | tr '\n' ' ' > "$OUT/rproc_after.txt"
dmesg | awk -v t0="$T0" '{ if (match($0, /^\[ *[0-9.]+\]/)) { ts = substr($0, 2, RLENGTH - 2) + 0; if (ts >= t0) print } }' > "$OUT/dmesg_new.txt"
tail -2 "$OUT/stdout.txt"; tail -2 "$OUT/stderr.txt"; echo "rc=$RC rproc=$(cat $OUT/rproc_after.txt) new_dmesg=$(wc -l < $OUT/dmesg_new.txt)"
