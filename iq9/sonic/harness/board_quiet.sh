#!/bin/sh
# board_quiet.sh -- sourced by the board scripts: require_quiet_board returns non-zero (and says why) when the IQ-9075 is busy:
# 1-min load average >= 1.0, another accelerator/deploy/simulator/stress process running, or a remote processor not "running".
# Measure only on a quiet board; a leftover process on an NSP or CPU changes every number.
require_quiet_board() {
  la=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$(awk -v l="$la" 'BEGIN{print (l>=1.0)}')" = 1 ]; then echo "busy: loadavg $la"; return 1; fi
  p=$(pgrep -f 'qnn-net-run|g1_deploy_onnx_ref|sonic_rt|sonic_bench|sim_driver.py|stress-ng' | grep -v "^$$\$" | head -3)
  if [ -n "$p" ]; then echo "busy: $(ps -o pid=,comm= -p $(echo $p | tr ' ' ',') 2>/dev/null | tr '\n' ' ')"; return 1; fi
  for s in /sys/class/remoteproc/*/state; do
    [ "$(cat $s)" = running ] || { echo "remoteproc $s = $(cat $s)"; return 1; }
  done
  return 0
}
