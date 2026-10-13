#!/usr/bin/env bash
# Jetson AGX Orin (JetPack 6): system python3 + user-site packages, TensorRT 10.3 from JetPack.
# Install the Jetson torch 2.8.0 / torchvision 0.23.0 wheels FIRST (docs/SETUP.md), then run this.
set -euo pipefail
cd "$(dirname "$0")"; source ./versions.env; export GROOT_PY="${GROOT_PY:-$(command -v python3)}"; source ./env.sh
python3 -c "import torch, tensorrt; assert torch.cuda.is_available(); print('torch', torch.__version__, 'trt', tensorrt.__version__)"
bash ./fetch_upstream.sh
pip3 install --user -r requirements-orin-jp6.txt
bash ./fetch_models.sh all
echo "Orin setup done. Run speed benchmarks with GROOT_PY=python3 (see docs/SPEED.md)."
