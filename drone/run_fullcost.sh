#!/usr/bin/env bash
# Full-cost decision rate + output gate on this machine, as published: one gate pass, then 3 full-cost runs of 64 decisions
# (+5 warm-up) at frame offsets 0 / 64 / 128, each in its own process. Prints the summary next to our numbers.
#   bash drone/run_fullcost.sh LABEL [rtx5090|thor|orin]           x86 / Orin
#   bash setup/thor_container.sh 'bash drone/run_fullcost.sh thor thor'   Jetson AGX Thor
# Pin clocks first if you can (Jetson: sudo nvpmodel -m 0 && sudo jetson_clocks; we did; the RTX 5090 was not locked).
# Outputs: $DRONE_RUNS/<LABEL>/{gate,full_r1,full_r2,full_r3}/
set -euo pipefail
cd "$(dirname "$0")/.."; source setup/env.sh; source drone/env.sh
L="${1:?label, e.g. mygpu}"; CMP="${2:-}"; PY="${GROOT_PY:-python3}"; O="$DRONE_RUNS/$L"; mkdir -p "$O"
"$PY" drone/drone_fulltrt.py gate --out "$O/gate" --tag "$L"
for r in 1 2 3; do
  "$PY" drone/drone_fulltrt.py fullcost --out "$O/full_r$r" --tag "${L}_r$r" --offset $(( (r - 1) * 64 )) --frames 64 --warmup 5
done
echo; echo "=== output gate ==="; "$PY" drone/score_gate.py "$O/gate"
echo; echo "=== full cost ==="
"$PY" drone/score_fullcost.py ${CMP:+--compare $CMP} "$O/full_r1" "$O/full_r2" "$O/full_r3"
