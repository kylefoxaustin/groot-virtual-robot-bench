#!/usr/bin/env bash
# Denoise-steps sweep (1..32 steps + 64-step reference) on the full-TensorRT path, two engine builds,
# plus the eager sweep for comparison. Published on the RTX 5090 only.
#   bash speed/run_denoise_sweep.sh ["clock note"]
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
NOTE="${1:-not recorded}"; PY="${GROOT_PY:-python3}"; OUT="$GVRB_WORK/speed/sweep"; mkdir -p "$OUT"
for b in A B; do
  E="$GVRB_WORK/speed/5090/eng$b"
  [ -f "$E/action_decoder.engine" ] || "$PY" "$GVRB_ROOT/speed/build_engines.py" "$E" 296
  "$PY" "$GVRB_ROOT/speed/denoise_sweep_fulltrt.py" "$E" "$OUT/denoise_sweep_$b.json"
done
CLOCKS_NOTE="$NOTE" "$PY" "$GVRB_ROOT/speed/make_sweep_record.py" "$OUT/denoise_sweep_A.json" "$OUT/denoise_sweep_B.json" \
  "$OUT/denoise_fulltrt.json" "$OUT/stagetime_fulltrt.json"
CLOCKS_NOTE="$NOTE" "$PY" "$GVRB_ROOT/speed/denoise_sweep_eager.py" --out "$OUT/denoise_eager.json"
"$PY" "$GVRB_ROOT/speed/stage_time_eager.py" "$OUT/stagetime_eager.json"
echo "records -> $OUT   (compare with results/speed/n15_5090_*.json)"
