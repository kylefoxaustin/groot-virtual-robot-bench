#!/bin/sh
# (board, as root) Where does a stand-in call's time go? No nav; the stand-in alone on NSP1, governor performance.
#  (a) EXECT N calls (per-stage wall + CPU timers: list read, input populate, execute, output write), then EXECT N calls under
#      `perf record -g` of the orchestrator (sampled call stacks; needs perf on the board), then EXECF N calls (timers)
#  (b) check mode with EXEC and with EXECF, outputs kept: compare them byte for byte (EXECF must write the same bytes)
# Default stand-in: the dilation-free seg_512.
#   sh drone/iq9/contention/profile_exec.sh TAG [N] [CTX_BIN] [GRAPH] [LIST]
set -u
. "${DRONE_IQ9_BOARD_DIR:?source the board env.sh}/env.sh"
TAG=${1:?tag}; N=${2:-30}; B=$DRONE_IQ9_BOARD_DIR
CTXB=${3:-$B/standins/ctx/seg512df_fix_fp16.bin}; GRAPH=${4:-seg512df}; LIST=${5:-$B/standins/seg_list.txt}
O=$B/runs/profile/$TAG; mkdir -p $O; PY=$DRONE_IQ9_PY; W=$B/drone/iq9/contention/npu_worker.py
if pgrep -f "resident_orch|qnn-net-run" > /dev/null; then echo "another NSP process is running" | tee $O/orch.log; exit 1; fi
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $p/scaling_governor; done
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "gov $p $(cat $p/scaling_governor) cur=$(cat $p/scaling_cur_freq)" >> $O/orch.log; done
C="--ctx $CTXB --graph $GRAPH --device 1 --list $LIST --warmup 3 --bin $B/bin/resident_orch_drone_g"
cd $B
$PY $W burst $C --cmd EXECT --spec n:$N --out $O/w_EXECT.json --tag pt < /dev/null >> $O/orch.log 2>&1
$PY $W burst $C --cmd EXECT --spec n:$N --out $O/w_EXECT_perf.json --tag pp < /dev/null >> $O/orch.log 2>&1 &
WP=$!
n=0; OP=""; until [ -n "$OP" ] || [ $n -gt 600 ]; do sleep 0.5; n=$((n+1)); OP=$(pgrep -P $WP -f "resident_orch_drone_g"); done
echo "perf pid $OP" >> $O/orch.log
[ -n "$OP" ] && perf record -g -F 999 -p $OP -o $O/perf.data -- sleep ${PERF_S:-8} >> $O/orch.log 2>&1
wait $WP
$PY $W burst $C --cmd EXECF --spec n:$N --out $O/w_EXECF.json --tag pf < /dev/null >> $O/orch.log 2>&1
$PY $W check $C --cmd EXEC --out $O/w_check_EXEC.json --keep $O/keep_EXEC --tag ce < /dev/null >> $O/orch.log 2>&1
$PY $W check $C --cmd EXECF --out $O/w_check_EXECF.json --keep $O/keep_EXECF --tag cf < /dev/null >> $O/orch.log 2>&1
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $p/scaling_governor; done
if [ -f $O/perf.data ]; then
  perf report -i $O/perf.data --no-children --sort symbol --stdio 2>/dev/null | head -60 > $O/perf_flat.txt
  perf report -i $O/perf.data --children --sort symbol --stdio -g none 2>/dev/null | head -60 > $O/perf_children.txt
fi
for f in $O/keep_EXEC/out_*.raw; do cmp -s $f $O/keep_EXECF/$(basename $f) && echo "same $(basename $f)" || echo "DIFFERENT $(basename $f)"; done | tee -a $O/orch.log
echo PROFILE_DONE >> $O/orch.log; echo "done: $O (stage timers: stage_us_mean in w_EXECT.json / w_EXECF.json)"
