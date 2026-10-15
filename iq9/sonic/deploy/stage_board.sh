#!/bin/bash
# (x86 host) Assemble the board tree in $IQ9_SONIC_WORK/board and copy it to $IQ9_HOST:$IQ9_SONIC_BOARD_DIR, then install the
# simulator's Python packages on the board (no network needed there). Run after encdec/export.sh, planner/split.sh,
# deploy/build_deploy.sh, encdec/build_bench.sh, rt/build_rt.sh and deploy/build_clayer.sh.
#   bin/      g1_deploy_onnx_ref_qnn, sonic_bench, sonic_rt              lib/   onnxruntime, libzmq, upstream's aarch64 libddsc/libddscxx
#   work/     the deploy's working dir: observation_config.yaml, reference/example/ (motions), placeholder ONNX files (the QNN shim
#             runs the context bins named in SONIC_QNN_*; the upstream planner ONNX is large and is not needed on the board)
#   ctx/      encoder/decoder context bins              planner/  host_H0..4 ONNX, manifest.txt, planner_{S,R,P,D}_aaf0_fp16.bin, calls
#   onnx/     upstream encoder/decoder ONNX (CPU control arm)
#   simsrc/   gear_sonic + unitree_sdk2py (from $SONIC_REPO), cyclonedds (+ _clayer), cyclone_home/lib/libddsc, sim_driver.py, mkref.py
#   simpy/    pip --target install of the simulator's wheels (on the board)       harness/, rt/: the board-side scripts
set -e
HERE=$(cd "$(dirname "$0")" && pwd); IS=$(cd "$HERE/.." && pwd); GVRB=$(cd "$IS/../.." && pwd)
W=$IQ9_SONIC_WORK; S=$W/board; B=$IQ9_SONIC_BOARD_DIR; UP=$SONIC_REPO; T=$SONIC_DEPLOY/thirdparty/unitree_sdk2/thirdparty/lib/aarch64
PYTAG=$(echo $IQ9_PYVER | tr -d .)
rm -rf $S; mkdir -p $S/{bin,lib,ctx,planner,onnx,harness,rt,data,simsrc/cyclone_home/lib} $S/work/policy/release $S/work/planner/target_vel/V2 $S/work/reference
cp $W/build/g1_deploy_onnx_ref_qnn $W/build/sonic_bench $W/build/sonic_rt $S/bin/
cp -a $IQ9_SONIC_ORT/lib/libonnxruntime.so* $IQ9_SONIC_SYSROOT/lib/libzmq.so* $T/* $S/lib/
cp $SONIC_DEPLOY/policy/release/observation_config.yaml $S/work/policy/release/
rsync -a $SONIC_DEPLOY/reference/example $S/work/reference/
for f in policy/release/model_decoder.onnx policy/release/model_encoder.onnx planner/target_vel/V2/planner_sonic.onnx; do
  echo "placeholder: the IQ-9075 QNN shim does not read this file; it runs the context bins named in SONIC_QNN_*" > $S/work/$f
done
cp $SONIC_DEPLOY/policy/release/model_encoder.onnx $SONIC_DEPLOY/policy/release/model_decoder.onnx $S/onnx/
cp $W/ctx/sonic_enc_core_aaf0_fp16.bin $W/ctx/sonic_dec_aaf0_fp16.bin $S/ctx/
cp $W/ctx/sonic_enc_core_fp16.bin $W/ctx/sonic_dec_fp16.bin $S/ctx/ 2>/dev/null || true   # optional default-fusion builds
P=$W/planner; cp $P/blocks/host_H*.onnx $P/blocks/manifest.txt $S/planner/; cp $P/blocks/host_H*.data $S/planner/ 2>/dev/null || true
cp $P/blocks/planner_calls.f32 $S/planner/ 2>/dev/null || echo "note: no planner_calls.f32 yet (planner/make_manifest.py with a refset)"
for b in S R P D; do cp $P/qnn/ctx/planner_${b}_aaf0_fp16.bin $S/planner/; done
rsync -a --exclude __pycache__ $UP/gear_sonic $S/simsrc/
rsync -a --exclude __pycache__ $UP/external_dependencies/unitree_sdk2_python/unitree_sdk2py $S/simsrc/
rsync -a --exclude __pycache__ --exclude '*.egg-info' $W/cyclonedds-0.10.2/cyclonedds $S/simsrc/
cp -a $T/libddsc.so* $S/simsrc/cyclone_home/lib/
printf 'in_wheel = False\nlibrary_path = "%s/simsrc/cyclone_home/lib/libddsc.so"\n' "$B" > $S/simsrc/cyclonedds/__library__.py
cp $GVRB/sonic/harness/sim_driver.py $GVRB/sonic/harness/mkref.py $S/simsrc/
cp $IS/harness/*.py $IS/harness/*.sh $IS/encdec/cpu_ort_run.py $S/harness/
cp $IS/rt/rt_run.sh $IS/rt/loaded_arms.sh $S/rt/
cat > $S/env.sh <<E
export IQ9_SONIC_BOARD_DIR=$B
export IQ9_QNN_ROOT=$IQ9_QNN_ROOT
export IQ9_PY_ROOT=$IQ9_PY_ROOT
E
# the simulator's wheels for the board Python (aarch64, cp$PYTAG), installed there with --target simpy
mkdir -p $S/wheels
python3 -m pip download --only-binary=:all: --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 --platform manylinux_2_28_aarch64 \
  --python-version $PYTAG --implementation cp -d $S/wheels mujoco==3.15.0 numpy==1.26.4 scipy==1.15.3 pyyaml==6.0.3 pyzmq==27.2.0 \
  msgpack==1.2.3 msgpack-numpy==0.4.8 opencv-python-headless==4.11.0.86 > $W/wheels_download.log 2>&1
ssh $IQ9_HOST "mkdir -p $B"
rsync -a $S/ $IQ9_HOST:$B/
ssh $IQ9_HOST ". $B/env.sh && cd $B && \$IQ9_PY_ROOT/bin/python3 -m pip install --no-index --find-links wheels --target simpy \
  mujoco==3.15.0 numpy==1.26.4 scipy==1.15.3 pyyaml==6.0.3 pyzmq==27.2.0 msgpack==1.2.3 msgpack-numpy==0.4.8 opencv-python-headless==4.11.0.86 | tail -2"
echo "staged -> $IQ9_HOST:$B"
