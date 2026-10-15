#!/bin/bash
# run_encdec.sh <encdec refset.npz>   (x86 host; drives the board over ssh) -- encoder/decoder correctness + latency on the IQ-9075.
# <refset.npz> = sonic/gate/make_refsets.py encdec <a reference GPU closed-loop run dir> (the exact tensors that run consumed/produced).
# Order matters: the fp16 prediction is written on the host BEFORE any board run on these inputs; the gate is derived from it.
# Board steps are quiet-gated and SIGTERM-only. Outputs: $IQ9_SONIC_WORK/encdec_test/.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}; NPZ=$(readlink -f "$1")
W=$IQ9_SONIC_WORK; R=$W/encdec_test; B=$IQ9_SONIC_BOARD_DIR; D=$B/data
mkdir -p $R/board $R/in $R/plant $R/wrongobs
sha256sum "$NPZ" | tee $R/npz.sha256

echo "== 1. fp16 prediction (host only, before any board run)"
$PY $HERE/fp16_predict.py "$NPZ" $W/onnx $R/pred.json | tee $R/pred.log
date -Iseconds > $R/pred.done_at

echo "== 2. board inputs: real, planted (reversed feature columns), wrong observation (features 64.. from 500 ticks away)"
$PY $HERE/npz_to_f32.py "$NPZ" $R/in
$PY $HERE/make_plant.py "$NPZ" $R/plant
$PY -c "
import numpy as np, sys
z = np.load(sys.argv[1]); d = z['dec_in'].astype(np.float32).copy(); d[:, 64:] = np.roll(z['dec_in'][:, 64:], 500, axis=0)
z['enc_in'].astype(np.float32).tofile(sys.argv[2] + '/enc_in.f32'); np.ascontiguousarray(d).tofile(sys.argv[2] + '/dec_in.f32')" "$NPZ" $R/wrongobs
ssh $IQ9_HOST "mkdir -p $D/encdec $D/encdec_plant $D/encdec_wrongobs $B/out/encdec"
rsync -a $R/in/ $IQ9_HOST:$D/encdec/; rsync -a $R/plant/ $IQ9_HOST:$D/encdec_plant/; rsync -a $R/wrongobs/ $IQ9_HOST:$D/encdec_wrongobs/

C=$B/ctx; O=$B/out/encdec; RUN=". $B/env.sh && sh $B/harness/board_run.sh"
NPU="--enc_kind core --mode correct --enc $C/sonic_enc_core_aaf0_fp16.bin --dec $C/sonic_dec_aaf0_fp16.bin"
echo "== 3. correctness on the board: NPU (aaf0), NPU planted, NPU wrong observation, CPU onnxruntime fp32"
ssh $IQ9_HOST "$RUN $O/npu_aaf0 $NPU --enc_in $D/encdec/enc_in.f32 --dec_in $D/encdec/dec_in.f32 | tail -1; sleep 3
  $RUN $O/npu_plant $NPU --enc_in $D/encdec_plant/enc_in.f32 --dec_in $D/encdec_plant/dec_in.f32 | tail -1; sleep 3
  $RUN $O/npu_wrongobs $NPU --enc_in $D/encdec_wrongobs/enc_in.f32 --dec_in $D/encdec_wrongobs/dec_in.f32 | tail -1; sleep 3
  . $B/env.sh && cd $B && timeout -s TERM 1800 \$IQ9_PY_ROOT/bin/python3 -I harness/cpu_ort_run.py --mode correct --enc onnx/model_encoder.onnx \
     --dec onnx/model_decoder.onnx --enc_in $D/encdec/enc_in.f32 --dec_in $D/encdec/dec_in.f32 --out $O/cpu_correct"
rsync -a $IQ9_HOST:$O/ $R/board/
for a in npu_aaf0:core:npu npu_plant:core:npu cpu_correct:full:cpu; do
  IFS=: read d k arm <<< "$a"
  $PY $HERE/gate.py $R/pred.json "$NPZ" $R/board/$d $k --arm $arm --label "$d" > /dev/null
  $PY -c "import json;g=json.load(open('$R/board/$d/gate.json'));print('$d',g['verdict'],g['board_enc'],{k:round(v,6) for k,v in g['board_dec'].items()})"
done
echo "   (npu_plant must FAIL; a gate that passes a planted fault is not checking anything)"

echo "== 4. latency on the board (aaf0): back-to-back, paced 50 Hz, paced 50 Hz without the NPU perf vote, profiled; CPU paced"
L="--enc $C/sonic_enc_core_aaf0_fp16.bin --dec $C/sonic_dec_aaf0_fp16.bin --enc_kind core --enc_in $D/encdec/enc_in.f32 --dec_in $D/encdec/dec_in.f32 --mode latency --warmup 100"
ssh $IQ9_HOST "$RUN $O/lat_npu_b2b $L --reps 5000 --period_us 0 | tail -1; sleep 3
  $RUN $O/lat_npu_paced50 $L --reps 3000 --period_us 20000 | tail -1; sleep 3
  $RUN $O/lat_npu_paced50_noperf $L --reps 3000 --period_us 20000 --perf none | tail -1; sleep 3
  $RUN $O/lat_npu_b2b_profiled $L --reps 3000 --period_us 0 --profile | tail -1; sleep 3
  . $B/env.sh && cd $B && timeout -s TERM 1800 \$IQ9_PY_ROOT/bin/python3 -I harness/cpu_ort_run.py --mode latency --enc onnx/model_encoder.onnx \
     --dec onnx/model_decoder.onnx --enc_in $D/encdec/enc_in.f32 --dec_in $D/encdec/dec_in.f32 --out $O/lat_cpu_paced50 --reps 3000 --warmup 100 --period_us 20000"
rsync -a $IQ9_HOST:$O/ $R/board/
$PY $HERE/lat_summary.py $R/board/lat_*
for d in $R/board/lat_*; do $PY $HERE/loop_check.py $R/pred.json $d > /dev/null && echo "$(basename $d) loop output check -> $d/loop_check.json"; done
echo "DONE -> $R"
