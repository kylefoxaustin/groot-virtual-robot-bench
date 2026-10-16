#!/bin/bash
# (x86 host) Assemble the drone's board tree in $DRONE_IQ9_WORK/board and copy it to $IQ9_HOST:$DRONE_IQ9_BOARD_DIR.
# Run after the steps that make its inputs (README); anything not built yet is skipped with a note, so you can re-run it.
#   drone/     drone_head_np.py, drone_metrics.py, assets/ (gate_set.npz, drone_head.npz) and iq9/ (these scripts)
#   bin/       resident_orch_drone, resident_orch_drone_g                     (port/build_resident_orch.sh)
#   ctx289/    vlln289_fix_fp16.bin, vlln289_aaf0_fp16.bin                     (port/build_vlln289.sh)
#   gate_frames/NNNN/{input_ids,attention_mask,pixel_values}.raw              (port/capture_gate_inputs.py)
#   standins/  ctx/*.bin (contention/build_standins_qnn.sh), the seeded check inputs as .raw, and input lists
#   groot/     Isaac-GR00T/ (python sources of $GROOT_REPO), shims/ (repo shims/jetson: aarch64 stubs), metadata.json
#              ($GROOT_MODEL/experiment_cfg), groot_step0.npz ($GROOT_STEP_NPZ): GR00T's transforms run on the board
#   env.sh     the board paths (source it in every board shell)
# The GR00T port's own files stay where iq9/README.md put them ($IQ9_WORK/optionB: ctx_fix, ctx_aaf0, glue_constants, io).
#   bash drone/iq9/stage_board.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd); DRONE=$(cd "$HERE/.." && pwd); GVRB=$(cd "$DRONE/.." && pwd)
W=$DRONE_IQ9_WORK; S=$W/board; B=$DRONE_IQ9_BOARD_DIR
[ "$IQ9_HOST" != "<board>" ] || { echo "set IQ9_HOST to the board's ssh target" >&2; exit 1; }
rm -rf $S; mkdir -p $S/drone/assets $S/bin $S/ctx289 $S/standins/ctx $S/groot
cp $DRONE/drone_head_np.py $DRONE/drone_metrics.py $S/drone/
cp $DRONE/assets/gate_set.npz $DRONE/assets/drone_head.npz $S/drone/assets/
rsync -a --exclude __pycache__ $HERE/ $S/drone/iq9/
note() { echo "note: $*"; }
cp $W/orch/build/resident_orch_drone $W/orch/build/resident_orch_drone_g $S/bin/ 2>/dev/null || note "no orchestrator yet (port/build_resident_orch.sh)"
cp $W/ctx289/vlln289_fix_fp16.bin $W/ctx289/vlln289_aaf0_fp16.bin $S/ctx289/ 2>/dev/null || note "no vlln289 bins yet (port/build_vlln289.sh)"
[ -f $W/gate_frames/manifest.json ] && rsync -a $W/gate_frames $S/ || note "no gate frames yet (port/capture_gate_inputs.py)"
cp $W/standins_ctx/*.bin $S/standins/ctx/ 2>/dev/null || note "no stand-in bins yet (contention/build_standins_qnn.sh)"
cp $DRONE_STANDINS/seg_tile*.raw $DRONE_STANDINS/radar_cloud*.raw $S/standins/ 2>/dev/null || note "no stand-in inputs (drone/contention/make_standins.py)"
cp $DRONE_STANDINS/seg1024/seg1024_tile*.raw $S/standins/ 2>/dev/null || note "no seg1024 tiles (contention/make_seg1024_check.py)"
( cd $S/standins; for p in seg:seg_tile radar:radar_cloud seg1024:seg1024_tile; do
    n=${p%%:*}; f=${p#*:}; ls $f*.raw > /dev/null 2>&1 && ls $f*.raw | sort -V | sed "s|^|$B/standins/|" > ${n}_list.txt; done )
rsync -a --exclude .git --exclude __pycache__ --exclude demo_data --exclude media $GROOT_REPO/ $S/groot/Isaac-GR00T/
rsync -a --exclude __pycache__ $GVRB/shims/jetson/ $S/groot/shims/
cp $GROOT_MODEL/experiment_cfg/metadata.json $S/groot/metadata.json
cp $GROOT_STEP_NPZ $S/groot/groot_step0.npz
cat > $S/env.sh <<E
export DRONE_IQ9_BOARD_DIR=$B
export DRONE_ASSETS=$B/drone/assets
export QAIRT_VER=$QAIRT_VER
export IQ9_WORK=$IQ9_WORK
export IQ9_QNN_ROOT=$IQ9_QNN_ROOT
export IQ9_PY_ROOT=$IQ9_PY_ROOT
export DRONE_IQ9_PY=$DRONE_IQ9_PY
export DRONE_IQ9_BLAS=$DRONE_IQ9_BLAS
export DRONE_IQ9_GLUE=$DRONE_IQ9_GLUE
export DRONE_IQ9_CONST=$DRONE_IQ9_CONST
export DRONE_IQ9_VIT_FIX=$DRONE_IQ9_VIT_FIX
export DRONE_IQ9_VIT_AAF0=$DRONE_IQ9_VIT_AAF0
export DRONE_IQ9_LLM=$DRONE_IQ9_LLM
export NO_ALBUMENTATIONS_UPDATE=1
E
ssh $IQ9_HOST "mkdir -p $B"
rsync -a $S/ $IQ9_HOST:$B/
echo "staged -> $IQ9_HOST:$B   (on the board: . $B/env.sh)"
