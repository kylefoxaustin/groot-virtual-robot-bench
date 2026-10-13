#!/usr/bin/env bash
# Sample Jetson BPMP actmon (DDR activity) + CPU% while one GR00T stage engine runs a sustained trtexec replay.
# Step 1 of the DDR-bandwidth measurement (results/profiling/groot_ddr_bandwidth_2026-09-24.json):
#   1. sudo bash actmon_stage.sh <stage> <engine>      x {vit, llm, dit}     -> actmon mean/sd + cpu% per stage
#   2. sudo python3 gpu_bw_cal.py <board>                                    -> actmon <-> GB/s calibration curve
#      (+ gpu_bw_multistream.py for the saturated points)
#   3. python3 derive_gbps.py <record.json>                                  -> GB/s per stage, % of MEASURED saturation
# actmon needs root (debugfs). jtop's EMC value is inert on JetPack 6/7 -- use this counter instead.
# STATUS: wrapper re-implemented from the record's documented method (trtexec --duration=12 --warmUp=2000, 0.5 s
# samples); the calibration and derivation scripts are the originals.
set -euo pipefail
STAGE="${1:?usage: actmon_stage.sh <stage> <engine>}"; ENGINE="${2:?usage: actmon_stage.sh <stage> <engine>}"; DUR="${DUR:-12}"
A=/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity
[ -r "$A" ] || { echo "need root: $A" >&2; exit 1; }
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
cpu(){ awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat; }
"$TRTEXEC" --loadEngine="$ENGINE" --duration="$DUR" --warmUp=2000 --noDataTransfers > "/tmp/actmon_$STAGE.log" 2>&1 &
PID=$!; sleep 3                                     # past warm-up
read b0 i0 < <(cpu); S=()
while kill -0 $PID 2>/dev/null; do S+=("$(cat $A)"); sleep 0.5; done
read b1 i1 < <(cpu)
rc=0; wait $PID || rc=$?
if [ $rc -ne 0 ]; then
  echo "ERROR: trtexec exited $rc for stage $STAGE (log: /tmp/actmon_$STAGE.log) -- samples discarded" >&2
  tail -5 "/tmp/actmon_$STAGE.log" >&2 || true
  exit $rc
fi
GPU_MS=$(grep -m1 "GPU Compute Time" "/tmp/actmon_$STAGE.log" | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
printf '%s\n' "${S[@]}" | python3 -c "
import sys, json, statistics as st
v=[int(x) for x in sys.stdin if x.strip()][:-1]          # drop the tail sample taken as trtexec exits
print(json.dumps({'stage':'$STAGE','actmon':int(st.mean(v)),'sd':int(st.pstdev(v)),'n':len(v),
  'cpu_pct':round(100*($b1-$b0)/max(1,($b1-$b0)+($i1-$i0)),1),'gpu_ms':float('${GPU_MS:-nan}')}))"
