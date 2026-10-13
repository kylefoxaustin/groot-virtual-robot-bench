#!/usr/bin/env bash
# The IQ-9075 scripts expect their sibling directories under $GVRB_WORK (optionB/, closed_loop_m2..m8/, qnn/).
# This symlinks iq9/work/* there, so the scripts run from the repo and their outputs land next to them.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${GVRB_WORK:?source setup/env.sh first}"
mkdir -p "$GVRB_WORK"
for d in "$here"/work/*; do
  n="$(basename "$d")"; t="$GVRB_WORK/$n"
  if [ -e "$t" ] && [ ! -L "$t" ]; then echo "skip $n: $t exists and is not a symlink"; continue; fi
  ln -sfn "$d" "$t"; echo "linked $t -> $d"
done
