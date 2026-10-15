#!/bin/bash
# Controlled band-drop test on the x86 + NVIDIA GPU host (our RTX 5090 runs): 18 unloaded runs bd_D<ms>_r<n> in the order of
# banddrop_order.json ("rtx5090"), 15 s apart, NVIDIA's stock deploy in its container. env from sonic/env.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=$SONIC_RUNS; mkdir -p $R
for r in $(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['rtx5090']))" $HERE/banddrop_order.json); do
  D=$((10#$(echo $r | sed 's/^bd_D\([0-9]*\)_.*/\1/')))
  python3 $HERE/run_closed_loop_x86.py --run_dir $R/$r --script v1 --release_delay_ms $D > $R/$r.orch 2>&1
  sleep 15
done
