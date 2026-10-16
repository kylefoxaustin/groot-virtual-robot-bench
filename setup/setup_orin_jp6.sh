#!/usr/bin/env bash
# Jetson AGX Orin (JetPack 6): system python3 + user-site packages, TensorRT 10.3 from JetPack.
# Install the Jetson torch 2.8.0 / torchvision 0.23.0 wheels FIRST (docs/SETUP.md), then run this.
#   GROOT_PY=python3 bash setup/setup_orin_jp6.sh [--policy-only]   (--policy-only: base model only, enough for speed and drone/)
set -euo pipefail
WHAT=all; [ "${1:-}" = --policy-only ] && WHAT=base
cd "$(dirname "$0")"; source ./versions.env; export GROOT_PY="${GROOT_PY:-$(command -v python3)}"; source ./env.sh
python3 -c "import torch, tensorrt; assert torch.cuda.is_available(); print('torch', torch.__version__, 'trt', tensorrt.__version__)"
bash ./fetch_upstream.sh
# numpydantic 1.10.0 declares numpy >= 2, but JetPack 6's TensorRT 10.3 and the Jetson torch wheel need numpy 1.x: install the
# rest first, then pin numpy (the same two-step the Thor container uses; pip warns about numpydantic, which works on 1.26.4).
grep -v '^numpy==' requirements-orin-jp6.txt > /tmp/gvrb-req-orin.txt
pip3 install --user -r /tmp/gvrb-req-orin.txt
pip3 install --user "$(grep '^numpy==' requirements-orin-jp6.txt)"
bash ./fetch_models.sh $WHAT
echo "Orin setup done. Run speed benchmarks with GROOT_PY=python3 (see docs/SPEED.md)."
