#!/usr/bin/env bash
# IQ-9075 (Hexagon v73) per-stage accelerator time for the three big GR00T graphs (ViT, LLM, one DiT step).
# Runs ON THE BOARD: qnn-net-run on each fp16 context binary with the captured real inputs and basic profiling,
# then reads "Backend (Accelerator (execute) time)" from qnn-profile-viewer -- the DSP's own execute time.
# That number is ACCELERATOR time, not wall time; the whole-policy wall time comes from the resident loop
# (iq9/work/optionB/resident/resident_gframe_loop.py).
#
# usage (on the IQ-9075):  IQ9_QNN_ROOT=/opt/qnn IQ9_WORK=/opt/gvrb bash stage_time_qnn.sh [N_ITERS]
#   expects  $IQ9_WORK/qnn/ctx/groot_{vit,llm,dit}_fp16.bin   (from iq9/work/optionB/build_groot_aaf0.sh or the qnn/ cfgs)
#            $IQ9_WORK/qnn/io/input_list_g{vit,llm,dit}.txt   (raw inputs captured by iq9/work/optionB/src/groot_capture_ref.py)
# Never SIGKILL qnn-net-run: a killed DSP session can leave the NSP degraded until reboot.
set -euo pipefail
: "${IQ9_QNN_ROOT:?set IQ9_QNN_ROOT to the QAIRT/QNN SDK on the board}"; : "${IQ9_WORK:?set IQ9_WORK}"
N="${1:-20}"
B="$IQ9_QNN_ROOT/bin/aarch64-oe-linux-gcc11.2"; [ -x "$B/qnn-net-run" ] || B="$IQ9_QNN_ROOT/bin"
L="$IQ9_QNN_ROOT/lib/aarch64-oe-linux-gcc11.2"; [ -d "$L" ] || L="$IQ9_QNN_ROOT/lib"
export LD_LIBRARY_PATH="$L:${LD_LIBRARY_PATH:-}" ADSP_LIBRARY_PATH="$IQ9_QNN_ROOT/lib/hexagon-v73/unsigned"
echo "{"
first=1
for g in vit llm dit; do
  out="$IQ9_WORK/qnn/out_stage_$g"; rm -rf "$out"
  "$B/qnn-net-run" --backend "$L/libQnnHtp.so" --retrieve_context "$IQ9_WORK/qnn/ctx/groot_${g}_fp16.bin" \
     --input_list "$IQ9_WORK/qnn/io/input_list_g${g}.txt" --output_dir "$out" --profiling_level basic \
     --perf_profile burst --num_inferences "$N" >"$out.log" 2>&1
  acc_us=$("$B/qnn-profile-viewer" --input_log "$out/qnn-profiling-data_0.log" 2>/dev/null \
           | grep -A12 "Execute Stats (Average)" | awk '/Backend \(Accelerator \(execute\) time\)/{print $(NF-1); exit}')
  [ -n "$acc_us" ] || { echo "no accelerator time for $g -- did the NPU run? see $out.log" >&2; exit 1; }
  [ $first = 1 ] || echo ","; first=0
  printf '  "%s_accel_ms": %.3f' "$g" "$(awk -v a="$acc_us" 'BEGIN{print a/1000}')"
done
echo; echo "}"
