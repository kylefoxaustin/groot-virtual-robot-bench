#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
set -euo pipefail
cd ${GVRB_WORK}
for spec in "ctrl 8" "a50 16" "a50 8" "a85 16" "a85 8"; do
  echo "@@@ START $spec $(date +%T)"
  bash closed_loop_m8/build_m8.sh $spec 2>&1 | tail -4
done
echo "ALL_M8_BUILDS_DONE $(date +%T)"
