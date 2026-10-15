#!/bin/bash
# (x86 host) Upstream encoder/decoder ONNX -> QNN-friendly graphs -> CPU equivalence check -> v73 fp16 context bins.
# Needs python3 with onnx + onnxruntime + numpy, and the QAIRT SDK. Outputs in $IQ9_SONIC_WORK/{onnx,dlc,cfg,ctx}.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}; W=$IQ9_SONIC_WORK; mkdir -p $W/onnx
$PY $HERE/make_graphs.py $SONIC_DEPLOY/policy/release $W/onnx
$PY $HERE/check_core_equiv.py $W                 # must print exact True: the split + host Round is the upstream encoder
bash $HERE/build_ctx.sh sonic_enc_core aaf0       # what the deploy shim and sonic_rt load
bash $HERE/build_ctx.sh sonic_dec aaf0
if [ "${WITH_DEFAULT_FUSION:-0}" = 1 ]; then      # optional: the same graphs with the default activation fusion, for comparison
  bash $HERE/build_ctx.sh sonic_enc_core; bash $HERE/build_ctx.sh sonic_dec
fi
