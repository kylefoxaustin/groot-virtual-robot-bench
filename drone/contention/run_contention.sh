#!/usr/bin/env bash
# The drone's flight brain (the nav policy, back to back on the 256 gate frames) sharing the computer with other AI jobs.
# Each job is a separate process on the same GPU (default time-slicing, no MPS), as published.
#   scan      : 5 bursts of the 50 MP inspection AI stand-in (seg_512 x 190 tiles), 5 s apart, after 6 s of nav alone
#   radar     : 5 windows of the radar fusion stand-in at 20 Hz for 4 s, then 5 windows of 400 back-to-back fusions
#               (that second arm turned out too short to score; kept as published)
#   radarflat : 5 windows of back-to-back radar fusion for 4 s each
#   all       : radar at 20 Hz + route planning (seg_1024, one frame every PLAN_PERIOD s, an ASSUMED duty) running
#               throughout, plus the 5 scan bursts; the nav is binned inside vs outside the bursts
#   bash drone/contention/run_contention.sh MODE TAG
# Then: python3 drone/contention/score_run.py $DRONE_RUNS/contention/TAG [--compare <published run name>]
# env:  NAV_IN_THOR_CONTAINER=1  start the nav loop inside setup/thor_container.sh (Jetson AGX Thor; workers stay native)
#       WORKER_PY  python for the workers (TensorRT 10.x + cuda-python); default $DRONE_WORK/venv-workers/bin/python on x86
#                  (bash drone/contention/setup_workers_x86.sh), python3 on a Jetson
#       NAV_CPUS / WORKER_CPUS  taskset CPU lists (optional);  DRONE_TORCH_THREADS (optional)
#       NBURST=5 SOLO_S=6 GAP_S=5 PLAN_PERIOD=3.0 DUR (nav seconds: scan 75, radar 125, radarflat 70, all 85)
# Pin clocks first (Jetson: sudo nvpmodel -m 0 && sudo jetson_clocks). Nothing here is killed; every process ends on its own.
set -euo pipefail
cd "$(dirname "$0")/../.."; source setup/env.sh; source drone/env.sh
MODE="${1:?scan|radar|radarflat|all}"; TAG="${2:?tag}"; O="$DRONE_RUNS/contention/$TAG"
mkdir -p "$O"; rm -f "$O"/ready "$O"/*.json "$O"/*started "$O"/orch.log
S="$DRONE_STANDINS"; SEG512="${SEG512:-$S/seg_512.engine}"; SEG1024="${SEG1024:-$S/seg_1024.engine}"; RADAR="${RADAR:-$S/radar_fusion.engine}"
NBURST=${NBURST:-5}; SOLO_S=${SOLO_S:-6}; GAP_S=${GAP_S:-5}; PLAN_PERIOD=${PLAN_PERIOD:-3.0}
case $MODE in scan) DUR=${DUR:-75};; radar) DUR=${DUR:-125};; radarflat) DUR=${DUR:-70};; all) DUR=${DUR:-85};; *) echo "bad mode" >&2; exit 1;; esac
if [ "$(uname -m)" = aarch64 ]; then WPY="${WORKER_PY:-python3}"; else WPY="${WORKER_PY:-$DRONE_WORK/venv-workers/bin/python}"; fi
[ -n "${WORKER_CPUS:-}" ] && WPY="taskset -c $WORKER_CPUS $WPY"
NAVPFX=""; [ -n "${NAV_CPUS:-}" ] && NAVPFX="taskset -c $NAV_CPUS"
for f in "$SEG512" "$SEG1024" "$RADAR"; do [ -f "$f" ] || { echo "missing $f -- drone/contention/build_engines.sh" >&2; exit 1; }; done
VW=drone/contention/vision_worker.py; PW=drone/contention/periodic_worker.py
log() { echo "$*" >> "$O/orch.log"; }
log "nav_cpus ${NAV_CPUS:-all} worker_cpus ${WORKER_CPUS:-all} torch_threads ${DRONE_TORCH_THREADS:-default}"
log "mode $MODE tag $TAG seg512 $SEG512 seg1024 $SEG1024 radar $RADAR plan_period $PLAN_PERIOD utc $(date -u +%FT%TZ)"
LOOP="drone/drone_fulltrt.py loop --out $O --tag $TAG --warmup 10 --duration $DUR --ready $O/ready"
if [ "${NAV_IN_THOR_CONTAINER:-0}" = 1 ]; then
  bash setup/thor_container.sh "${DRONE_TORCH_THREADS:+DRONE_TORCH_THREADS=$DRONE_TORCH_THREADS} python $LOOP" > "$O/nav.log" 2>&1 &
else
  $NAVPFX "${GROOT_PY:-python3}" $LOOP > "$O/nav.log" 2>&1 &
fi
NAV=$!
n=0; until [ -f "$O/ready" ]; do sleep 0.5; n=$((n+1))
  [ $n -gt 1800 ] && { log "nav never ready"; exit 1; }
  kill -0 $NAV 2>/dev/null || { log "nav died"; tail -5 "$O/nav.log"; exit 1; }; done
log "nav ready $(cat "$O/ready")"
sleep "$SOLO_S"
scan_bursts() {
  for r in $(seq 1 "$NBURST"); do
    log "burst $r spawn $(date +%s.%N)"; $WPY $VW "$SEG512" n:190 "$O/vis_burst$r.json" >> "$O/orch.log" 2>&1
    log "burst $r exit $(date +%s.%N)"; sleep "$GAP_S"
  done
}
case $MODE in
scan) scan_bursts ;;
radar)
  for r in $(seq 1 "$NBURST"); do
    log "radar20 $r spawn $(date +%s.%N)"; $WPY $PW "$RADAR" hz:20,t:4.0 "$O/radar20_win$r.json" >> "$O/orch.log" 2>&1; log "radar20 $r exit $(date +%s.%N)"; sleep "$GAP_S"
  done
  for r in $(seq 1 "$NBURST"); do
    log "radarflat $r spawn $(date +%s.%N)"; $WPY $PW "$RADAR" flat:400 "$O/radarflat_win$r.json" >> "$O/orch.log" 2>&1; log "radarflat $r exit $(date +%s.%N)"; sleep "$GAP_S"
  done ;;
radarflat)
  for r in $(seq 1 "$NBURST"); do
    log "radarflat $r spawn $(date +%s.%N)"; $WPY $PW "$RADAR" hz:1000000,t:4.0 "$O/radarflat_win$r.json" >> "$O/orch.log" 2>&1; log "radarflat $r exit $(date +%s.%N)"; sleep "$GAP_S"
  done ;;
all)
  BG=${BG_S:-50}
  log "bg spawn $(date +%s.%N)"
  $WPY $PW "$RADAR" hz:20,t:$BG "$O/bg_radar.json" "$O/bg_radar.started" >> "$O/orch.log" 2>&1 & RP=$!
  $WPY $PW "$SEG1024" hz:$(python3 -c "print(1/$PLAN_PERIOD)"),t:$BG "$O/bg_plan.json" "$O/bg_plan.started" >> "$O/orch.log" 2>&1 & PP=$!
  n=0; until [ -f "$O/bg_radar.started" ] && [ -f "$O/bg_plan.started" ]; do sleep 0.2; n=$((n+1)); [ $n -gt 300 ] && { log "bg never started"; break; }; done
  log "bg started $(date +%s.%N)"
  sleep 6; scan_bursts
  wait $RP $PP; log "bg exit $(date +%s.%N)" ;;
esac
wait $NAV; log "nav rc $?"
if [ "$MODE" = scan ] || [ "$MODE" = all ]; then $WPY $VW "$SEG512" n:190 "$O/vis_alone.json" >> "$O/orch.log" 2>&1; fi
if [ "$MODE" = radar ] || [ "$MODE" = radarflat ]; then $WPY $PW "$RADAR" hz:1000000,t:4.0 "$O/radar_alone.json" >> "$O/orch.log" 2>&1; fi
log DONE
echo "done: python3 drone/contention/score_run.py $O"
