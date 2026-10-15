#!/bin/bash
# The IQ-9075 closed-loop runs (ON the board, as root, detached; from $IQ9_SONIC_BOARD_DIR with env.sh sourced). Graceful stops only.
#   iq9_runs.sh smoke                        one short run (stand, band drop, stop): check the whole chain first
#   iq9_runs.sh unloaded <first> <last> [fix] cl_r<first>..cl_r<last> (fix_cl_r* with the late-start settings), back-to-back
#   iq9_runs.sh cpu [fix]                     a4cpu_r1..r3: one stress-ng matrixprod hog per core, started 5 s before each run
#   iq9_runs.sh cpuarm                        cpuarm_r1: encoder+decoder on the CPU (onnxruntime fp32) instead of the NPU (control arm)
#   iq9_runs.sh diag                          diag_nonroot_cpu: the cpu arm with the deploy's SCHED_FIFO request disabled (n=1)
#   iq9_runs.sh banddrop <seed> [fix]         bd_D<ms>_r<n>: controlled band drop, D in {10,25,40,60,80,100} ms x 3 runs, unloaded,
#                                             in random.Random(<seed>).shuffle order (pick and record the seed before the first run)
# "fix" = the IQ-9075 late-start settings: --deploy_env SONIC_CTRL_FIFO=90,SONIC_QNN_PLANNER_THREADS=1,SONIC_MAIN_SLEEP=abs (control
# thread SCHED_FIFO 90, 1 planner onnxruntime thread, sleeping main loop) and the performance cpufreq governor for the batch,
# restored on exit. Without "fix" the deploy runs with upstream's scheduling (the latefix code is inert when the variables are unset).
# Every run is refused on a busy board (harness/board_quiet.sh) and runs 15 s after the previous one.
set -u
B=$IQ9_SONIC_BOARD_DIR; H=$B/harness; R=${SONIC_RUNS:-$B/runs}; mkdir -p $R; N=$(nproc); PY=$IQ9_PY_ROOT/bin/python3
. $H/board_quiet.sh
FIXENV="--deploy_env SONIC_CTRL_FIFO=90,SONIC_QNN_PLANNER_THREADS=1,SONIC_MAIN_SLEEP=abs"
gov_save() { for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo "$g $(cat $g)"; done > $R/governor_saved.txt
             trap 'while read g v; do echo $v > $g; done < $R/governor_saved.txt; echo "$(date -Iseconds) governor restored" >> $R/batch.log' EXIT; }
gov_perf() { gov_save; for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > $g; done
             echo "$(date -Iseconds) governor -> performance" >> $R/batch.log; }
run_cl() {   # run_cl <name> [orchestrator args...]
  local r=$1; shift; sleep 15
  if ! require_quiet_board >> $R/batch.log 2>&1; then echo "$(date -Iseconds) REFUSING $r: board busy" >> $R/batch.log; return 3; fi
  timeout -s TERM 1200 $PY $H/run_closed_loop_iq9.py --run_dir $R/$r --script ${SCRIPT:-v1} "$@" > $R/$r.orch 2>&1
  echo "rc=$?" >> $R/$r.orch
  echo "$(date -Iseconds) done $r $(tail -2 $R/$r.orch | tr '\n' ' ') rproc=$(cat /sys/class/remoteproc/*/state | tr '\n' ' ')" >> $R/batch.log
}
run_loaded() {   # run_loaded <name> [orchestrator args...]: CPU saturated (the hogs bypass the quiet check)
  local r=$1; shift; sleep 15
  stress-ng --cpu $N --cpu-method matrixprod --timeout 400s --metrics-brief > $R/$r.stress.log 2>&1 &
  local SP=$!; sleep 5; cat /proc/loadavg > $R/$r.loadavg_during
  timeout -s TERM 1200 $PY $H/run_closed_loop_iq9.py --run_dir $R/$r --script v1 "$@" > $R/$r.orch 2>&1
  echo "rc=$?" >> $R/$r.orch
  kill -TERM $SP; wait $SP
  echo "$(date -Iseconds) done $r (CPU saturated) $(tail -2 $R/$r.orch | tr '\n' ' ')" >> $R/batch.log
}
echo "$(date -Iseconds) start $*" >> $R/batch.log
case $1 in
smoke)    SCRIPT=smoke run_cl smoke_r1 ;;
unloaded) P=cl; X=""; [ "${4:-}" = fix ] && { P=fix_cl; X=$FIXENV; gov_perf; }
          for i in $(seq $2 $3); do run_cl ${P}_r$i $X; done ;;
cpu)      P=a4cpu; X=""; [ "${2:-}" = fix ] && { P=fix_a4cpu; X=$FIXENV; gov_perf; }
          for i in 1 2 3; do run_loaded ${P}_r$i $X; done ;;
cpuarm)   run_cl cpuarm_r1 --cpu_arm ;;
diag)     run_loaded diag_nonroot_cpu --deploy_prefix 'setpriv --bounding-set=-sys_nice --inh-caps=-sys_nice --' ;;
banddrop) X=""; [ "${3:-}" = fix ] && { X=$FIXENV; gov_perf; }
          ORDER=$($PY -c "import random,sys; L=['D%03d_r%d' % (D, r) for D in (10, 25, 40, 60, 80, 100) for r in (1, 2, 3)]; random.Random(int(sys.argv[1])).shuffle(L); print(' '.join(L))" $2)
          echo "$(date -Iseconds) banddrop seed=$2 order=$ORDER" >> $R/batch.log
          for r in $ORDER; do D=$((10#$(echo $r | sed 's/^D\([0-9]*\)_.*/\1/'))); run_cl bd_$r --release_delay_ms $D $X; done ;;
*) echo "usage: $0 smoke | unloaded <first> <last> [fix] | cpu [fix] | cpuarm | diag | banddrop <seed> [fix]"; exit 1 ;;
esac
echo "$(date -Iseconds) end $1" >> $R/batch.log
