#!/bin/bash
# Wait until no other GPU work is running for 6 consecutive 30 s polls (3 min sustained), then
# build + time + gate GR00T on Thor (thor_groot.sh, thor_gate.py, thor_summarize.py from this directory).
set -euo pipefail
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
HERE="$(cd "$(dirname "$0")" && pwd)"
clear=0
while [ $clear -lt 6 ]; do
  busy=0
  pgrep -x ncu >/dev/null && busy=1
  pgrep -f TreeLauncherSubreaper >/dev/null && busy=1
  pgrep -x trtexec >/dev/null && busy=1
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && busy=1
  if [ $busy -eq 1 ]; then clear=0; else clear=$((clear+1)); fi
  sleep 30
done
echo "GPU CLEAR (3 min sustained) at $(date) -- starting GR00T builds"
bash "$HERE/thor_groot.sh"
python3 "$HERE/thor_gate.py"
python3 "$HERE/thor_summarize.py" > "${GVRB_WORK}/thor_summary.txt" 2>&1
echo ALL_DONE
