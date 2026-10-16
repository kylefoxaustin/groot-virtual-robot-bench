#!/bin/sh
# (board, as root) Full-cost runs of the drone policy on the IQ-9075, one process per run, one model load at a time:
#   single : arms fix and aaf0 on NSP0 (device 0), 3 runs each at frame offsets 0 / 64 / 128, 64 decisions + 3 warm-up;
#            one fix run on NSP1 (device 1)
#   dual   : two streams at once, fix on NSP0 (offset 0) and NSP1 (offset 128), released together by a GO file; 3 repeats
# CPU governor set to performance for the batch (read back into the log) and restored to schedutil at the end.
# Before each run it waits for the 1-min load average to fall below 1 and refuses to start while another process holds
# an NSP. Nothing is killed; every process ends on its own.
#   sh drone/iq9/policy/run_fullcost_iq9.sh [OUT_DIR]      (default $DRONE_IQ9_BOARD_DIR/runs/fullcost)
# Then, on the host after copying OUT_DIR back: python3 drone/iq9/policy/score_fullcost_iq9.py OUT_DIR/s_fix_d0_r*
set -u
. "${DRONE_IQ9_BOARD_DIR:?source the board env.sh}/env.sh"
O=${1:-$DRONE_IQ9_BOARD_DIR/runs/fullcost}; mkdir -p $O
cd $DRONE_IQ9_BOARD_DIR
PY=$DRONE_IQ9_PY; RUN="drone/iq9/policy/drone_bframe_board.py fullcost"
LOG=$O/batch.log
echo "start $(date -u +%FT%TZ)" > $LOG
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $p/scaling_governor; done
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$p $(cat $p/scaling_governor) cur=$(cat $p/scaling_cur_freq) max=$(cat $p/cpuinfo_max_freq)" >> $LOG; done
quiet() {   # our own previous run inflates the 1-min load average: wait for it to decay; never share an NSP by accident
  n=0; while [ "$(awk '{print ($1<1.0)}' /proc/loadavg)" != 1 ] && [ $n -lt 60 ]; do sleep 5; n=$((n+1)); done
  echo "loadavg $(cut -d' ' -f1 /proc/loadavg)" >> $LOG
  if pgrep -f "resident_orch|qnn-net-run" > /dev/null; then echo "another NSP process is running: $(pgrep -af 'resident_orch|qnn-net-run')" >> $LOG; return 1; fi
}
single() {  # arm device offset tag
  quiet || { echo "NOT QUIET, skip $4" >> $LOG; return; }
  $PY $RUN --out $O/$4 --tag $4 --arm $1 --device $2 --offset $3 --frames 64 --warmup 3 >> $LOG 2>&1
}
for r in 1 2 3; do
  off=$(( (r-1)*64 ))
  single fix 0 $off s_fix_d0_r$r
  single aaf0 0 $off s_aaf0_d0_r$r
done
single fix 1 0 s_fix_d1_r1
for r in 1 2 3; do
  quiet || { echo "NOT QUIET, skip dual r$r" >> $LOG; continue; }
  GO=/dev/shm/drone_go_r$r; rm -f $GO
  $PY $RUN --out $O/d_fix_r${r}_d0 --tag d_fix_r${r}_d0 --arm fix --device 0 --offset 0 --frames 64 --warmup 3 --go-file $GO > $O/d_r${r}_d0.log 2>&1 &
  P0=$!
  $PY $RUN --out $O/d_fix_r${r}_d1 --tag d_fix_r${r}_d1 --arm fix --device 1 --offset 128 --frames 64 --warmup 3 --go-file $GO > $O/d_r${r}_d1.log 2>&1 &
  P1=$!
  n=0; until grep -q READY $O/d_r${r}_d0.log && grep -q READY $O/d_r${r}_d1.log; do sleep 1; n=$((n+1)); [ $n -gt 600 ] && break; done
  touch $GO; echo "dual r$r GO $(date -u +%FT%TZ)" >> $LOG
  wait $P0 $P1
  cat $O/d_r${r}_d0.log $O/d_r${r}_d1.log | grep -E "DONE|Error|ERR" >> $LOG
  rm -f $GO
done
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $p/scaling_governor; done
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "restored $p $(cat $p/scaling_governor)" >> $LOG; done
echo "BATCH_DONE $(date -u +%FT%TZ)" >> $LOG
