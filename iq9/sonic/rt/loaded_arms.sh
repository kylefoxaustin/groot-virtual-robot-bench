#!/bin/sh
# loaded_arms.sh <outroot> [fix]   (ON the board as root, env.sh sourced; detach it, e.g. nohup ... &)
# The SONIC 50 Hz encoder/decoder loop (sonic_rt --mode loaded, NSP0) under five conditions, 10000 paced ticks each:
#   a  baseline                  c  + planner at 10 Hz on the SAME NSP (deviceId 0)
#   d  + planner on the OTHER NSP (deviceId 1)       b  CPU saturated (stress-ng, one matrixprod hog per core)
#   e  planner on the other NSP + CPU saturated
# fix = the IQ-9075 late-start settings: control thread SCHED_FIFO 90 (SONIC_CTRL_FIFO=90), 1 planner ORT thread, performance
# governor for the batch (restored on exit). Score with rt/gaps.py: compute miss = step > 20 ms; late start = start-to-start gap
# > 21 ms; slot miss = gap > 25 ms.
R=$1; FIX=$2; mkdir -p $R; B=$IQ9_SONIC_BOARD_DIR; N=$(nproc)
D=$B/data/encdec; C="--enc $B/ctx/sonic_enc_core_aaf0_fp16.bin --dec $B/ctx/sonic_dec_aaf0_fp16.bin --enc_in $D/enc_in.f32 --dec_in $D/dec_in.f32 --mode loaded --reps 10000 --warmup 100 --period_us 20000 --encdec_device 0"
PT=4; KN=""
if [ "$FIX" = fix ]; then PT=1; KN="SONIC_CTRL_FIFO=90"
  for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo "$g $(cat $g)"; done > $R/governor_saved.txt
  trap 'while read g v; do echo $v > $g; done < $R/governor_saved.txt' EXIT
  for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > $g; done
fi
P="--planner --planner_dir $B/planner --calls $B/planner/planner_calls.f32 --planner_period_us 100000 --planner_threads $PT"
run() { n=$1; shift; head -1 /proc/stat > $R/$n.stat0; env $KN "$@"; head -1 /proc/stat > $R/$n.stat1; sleep 10; }
date -Iseconds > $R/started
run a sh $B/rt/rt_run.sh $R/a 600 $C
run c sh $B/rt/rt_run.sh $R/c 600 $C $P --planner_device 0
run d sh $B/rt/rt_run.sh $R/d 600 $C $P --planner_device 1
stress-ng --cpu $N --cpu-method matrixprod --timeout 270s > $R/b_stress.log 2>&1 &
sleep 5; run b env ALLOW_LOAD=1 sh $B/rt/rt_run.sh $R/b 600 $C
wait; sleep 10
stress-ng --cpu $N --cpu-method matrixprod --timeout 270s > $R/e_stress.log 2>&1 &
sleep 5; run e env ALLOW_LOAD=1 sh $B/rt/rt_run.sh $R/e 600 $C $P --planner_device 1
wait
date -Iseconds > $R/done
