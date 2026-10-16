#!/bin/sh
# (board, as root) A contention session on the IQ-9075: the drone nav (nav_loop.py, the full-cost decision path) back to back
# on NSP0 for the whole session, while co-tenant stand-ins run on the NPU from a PLAN file (plans/*.txt), one directive a line:
#   mode M                     label the session (scan / all = the GPU modes, so drone/contention/score_run.py can score it)
#   sleep S                    wait S seconds (nav alone, gaps, recovery)
#   run LABEL <worker args>    foreground npu_worker.py; its JSON is $O/LABEL.json
#   runpin CPUS LABEL <args>   as run, under taskset -c CPUS
#   bg LABEL <worker args>     background worker; waits until it has loaded; it ends at its own t or at the next stopbg
#   stopbg                     stop every bg worker (stop file), wait for them
# Worker args are expanded by the shell, with these set: $CTX (stand-in context bins), $IN (input lists), $ORCH and $ORCH_G
# (the two orchestrator builds). Name workers vis_burst1..N (scan bursts) and bg_radar / bg_plan (background) to match the
# GPU runs' windows; any other names are binned with bin_session.py and a windows file.
# env: NAV_CPUS = CPU list for the nav process (taskset; its orchestrator and BLAS threads inherit it), NAV_DEVICE (default 0),
#      NAV_ARM (fix). CPU governor performance for the session (read back into orch.log), schedutil restored at the end.
# The board must be quiet: it refuses to start while another process holds an NSP. No SIGKILL anywhere: workers QUIT their
# orchestrator (SIGTERM only on a hang), the nav ends at its stop file.
#   sh drone/iq9/contention/run_session.sh TAG PLAN_FILE
# Then copy $DRONE_IQ9_BOARD_DIR/runs/contention/TAG to the host and score it there (README, step 6).
set -u
. "${DRONE_IQ9_BOARD_DIR:?source the board env.sh}/env.sh"
TAG=${1:?tag}; PLAN=${2:?plan file}
B=$DRONE_IQ9_BOARD_DIR; O=$B/runs/contention/$TAG; mkdir -p $O; rm -f $O/ready $O/stop_nav $O/stop_bg $O/*.started
PY=$DRONE_IQ9_PY; C=$B/drone/iq9/contention
CTX=$B/standins/ctx; IN=$B/standins; ORCH=$B/bin/resident_orch_drone; ORCH_G=$B/bin/resident_orch_drone_g
cp $PLAN $O/plan.txt
log() { echo "$*" >> $O/orch.log; }
log "tag $TAG plan $PLAN nav_cpus ${NAV_CPUS:-all} nav_device ${NAV_DEVICE:-0} utc $(date -u +%FT%TZ)"
while read -r CMD LABEL REST; do   # expand every worker line now: an unset variable stops here, before anything runs
  case "$CMD" in run|runpin|bg) ( eval "set -- $REST" ) 2>/dev/null || { log "plan line does not expand: $CMD $LABEL $REST"; echo "plan line does not expand (unset variable?): $CMD $LABEL $REST" >&2; exit 1; } ;; esac
done < $PLAN
if pgrep -f "resident_orch|qnn-net-run" > /dev/null; then log "NOT QUIET: $(pgrep -af 'resident_orch|qnn-net-run')"; echo "another NSP process is running" >&2; exit 1; fi
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $p/scaling_governor; done
for p in /sys/devices/system/cpu/cpufreq/policy*; do log "gov $p $(cat $p/scaling_governor) cur=$(cat $p/scaling_cur_freq) max=$(cat $p/cpuinfo_max_freq)"; done
(uptime; cat /proc/loadavg; for r in /sys/class/remoteproc/remoteproc*; do echo "$(cat $r/name) $(cat $r/state)"; done; ps -eo pid,stat,pcpu,comm --sort=-pcpu | head -8) > $O/census_pre.txt 2>&1
cd $B
NT=""; [ -n "${NAV_CPUS:-}" ] && NT="taskset -c $NAV_CPUS"
$NT $PY $C/nav_loop.py --out $O --device ${NAV_DEVICE:-0} --arm ${NAV_ARM:-fix} --stop $O/stop_nav > $O/nav.log 2>&1 &
NAV=$!
n=0; until [ -f $O/ready ]; do sleep 1; n=$((n+1)); [ $n -gt 900 ] && { log "nav never ready"; break; }; kill -0 $NAV 2>/dev/null || { log "nav died"; tail -5 $O/nav.log; exit 1; }; done
log "nav ready $(cat $O/ready)"
BG=""
while read -r CMD LABEL REST; do
  case "$CMD" in
    mode) log "mode $LABEL" ;;
    sleep) log "sleep $LABEL $(date +%s.%N)"; sleep $LABEL ;;
    run) eval "set -- $REST"; log "arm $LABEL spawn $(date +%s.%N)"
         $PY $C/npu_worker.py "$@" --out $O/$LABEL.json --tag $LABEL < /dev/null >> $O/orch.log 2>&1
         log "arm $LABEL exit $(date +%s.%N) rc $?" ;;
    runpin) CPUS=$LABEL; eval "set -- $REST"; LABEL=$1; shift; log "arm $LABEL spawn $(date +%s.%N) pinned $CPUS"
         taskset -c $CPUS $PY $C/npu_worker.py "$@" --out $O/$LABEL.json --tag $LABEL < /dev/null >> $O/orch.log 2>&1
         log "arm $LABEL exit $(date +%s.%N) rc $?" ;;
    bg) eval "set -- $REST"; log "bg spawn $(date +%s.%N) $LABEL"
        $PY $C/npu_worker.py "$@" --stop $O/stop_bg --started $O/$LABEL.started --out $O/$LABEL.json --tag $LABEL < /dev/null >> $O/orch.log 2>&1 &
        BG="$BG $!"
        n=0; until [ -f $O/$LABEL.started ]; do sleep 0.2; n=$((n+1)); [ $n -gt 3000 ] && { log "bg $LABEL never started"; break; }; done ;;
    stopbg) touch $O/stop_bg; for b in $BG; do wait $b; log "bg pid $b exit rc $?"; done; BG=""; rm -f $O/stop_bg; log "stopbg done $(date +%s.%N)" ;;
    ''|\#*) ;;
    *) log "unknown plan directive: $CMD" ;;
  esac
done < $PLAN
[ -n "$BG" ] && { touch $O/stop_bg; for b in $BG; do wait $b; done; rm -f $O/stop_bg; }
touch $O/stop_nav; wait $NAV; log "nav rc $?"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $p/scaling_governor; done
for p in /sys/devices/system/cpu/cpufreq/policy*; do log "restored $p $(cat $p/scaling_governor)"; done
(uptime; ps -eo pid,stat,pcpu,comm | grep -E "resident|python"; for r in /sys/class/remoteproc/remoteproc*; do echo "$(cat $r/name) $(cat $r/state)"; done) > $O/census_post.txt 2>&1
log SESSION_DONE
echo "done: $O"
