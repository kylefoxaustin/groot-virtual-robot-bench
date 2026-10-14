#!/bin/bash
# The Jetson closed-loop campaign, as we ran it (run as root, detached; env from sonic/env.sh). Graceful stops only.
#   jetson_runs.sh unloaded <first> <last>   cl_r<first>..cl_r<last>: unloaded v1 runs, back-to-back (we ran 1-3, then 4-10)
#   jetson_runs.sh cpu                       a4cpu_r1..r3: one stress-ng matrixprod hog per core, started 5 s before each run
#   jetson_runs.sh gpu <orin|thor>           a4gpu_r1..r3: GR00T N1.5 full TensorRT back-to-back on the same GPU (60 s alone first)
#   jetson_runs.sh diag                      diag_nonroot_cpu: the cpu arm once with the deploy's SCHED_FIFO request disabled (n=1)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=$SONIC_RUNS; mkdir -p $R; N=$(nproc)
run_cl() { $SONIC_SIM_PY $HERE/run_closed_loop_jetson.py --run_dir $R/$1 --script v1 "${@:2}" > $R/$1.orch 2>&1; }
case $1 in
unloaded)
  for i in $(seq $2 $3); do run_cl cl_r$i; sleep 15; done ;;
cpu)
  for i in 1 2 3; do r=a4cpu_r$i
    stress-ng --cpu $N --cpu-method matrixprod --timeout 400s --metrics-brief > $R/$r.stress.log 2>&1 &
    SP=$!; sleep 5; cat /proc/loadavg > $R/$r.loadavg_during
    run_cl $r
    kill -TERM $SP; wait $SP; sleep 15
  done ;;
gpu)
  # GR00T co-tenant: needs the repo's GR00T setup (setup/env.sh) and full-TensorRT engines from speed/build_engines.py in $GROOT_ENGINES
  G=$R/a4gpu_groot.json; rm -f $G $G.ready $G.stop
  if [ "$2" = thor ]; then nohup bash $HERE/groot_cotenant_thor.sh $G 1500 > $R/a4gpu_groot.log 2>&1 &
  else sudo -E -u "${SUDO_USER:-$USER}" nohup bash $HERE/groot_cotenant_orin.sh $G 1500 > $R/a4gpu_groot.log 2>&1 & fi
  until [ -f $G.ready ]; do sleep 2; done
  sleep 60                                   # GR00T alone, same session
  for i in 1 2 3; do run_cl a4gpu_r$i; sleep 15; done
  sleep 30; touch $G.stop                    # graceful stop of the GR00T loop (never SIGKILL)
  until [ -f $G ]; do sleep 2; done ;;
diag)
  r=diag_nonroot_cpu
  stress-ng --cpu $N --cpu-method matrixprod --timeout 400s --metrics-brief > $R/$r.stress.log 2>&1 &
  SP=$!; sleep 5
  run_cl $r --deploy_prefix 'setpriv --bounding-set=-sys_nice --inh-caps=-sys_nice --'
  kill -TERM $SP; wait $SP ;;
*) echo "usage: $0 unloaded <first> <last> | cpu | gpu <orin|thor> | diag"; exit 1 ;;
esac
[ -n "${SUDO_UID:-}" ] && chown -R $SUDO_UID:${SUDO_GID:-$SUDO_UID} $R
date > $R/$1.alldone
