#!/bin/bash
# run_planner_gate.sh <planner refset.npz> <pred.json>   (x86 host; drives the board over ssh)
# Planner correctness on the IQ-9075: every logged call through the split planner on NSP1 (blocks dumped), and once more with a
# planted fault in block P; then on the host: the gate (gate.py, thresholds in pred.json written BEFORE this run), a per-block
# check of the board's block outputs against onnxruntime fp32 on the same block inputs (blkcheck.py: this is what localises a
# block the NPU computes wrong), and a pipeline-integrity check (integrity.py). Run from anywhere; works in $IQ9_SONIC_WORK/planner.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}; REF=$(readlink -f "$1"); PRED=$(readlink -f "$2")
W=$IQ9_SONIC_WORK/planner; B=$IQ9_SONIC_BOARD_DIR; O=$B/out/planner; cd $W; mkdir -p runs
RT=". $B/env.sh && sh $B/rt/rt_run.sh"
A="--mode planner_correct --planner_dir $B/planner --calls $B/planner/planner_calls.f32 --planner_device 1"
ssh $IQ9_HOST "mkdir -p $O; $RT $O/correct_nsp1 900 $A --dump_blocks | tail -1; sleep 5; $RT $O/plant_nsp1 900 $A --plant P | tail -1"
rsync -a $IQ9_HOST:$O/ runs/
$PY $HERE/gate.py runs/correct_nsp1 "$REF" "$PRED"
$PY $HERE/gate.py runs/plant_nsp1 "$REF" "$PRED"      # must FAIL
$PY $HERE/blkcheck.py runs/correct_nsp1 | tee runs/correct_nsp1/blkcheck.txt
$PY $HERE/integrity.py runs/correct_nsp1 "$REF" | tee runs/correct_nsp1/integrity.txt
