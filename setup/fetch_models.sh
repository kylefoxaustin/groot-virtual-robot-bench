#!/usr/bin/env bash
# Download the two public checkpoints at the exact revisions used (we never redistribute weights).
# Both are under NVIDIA's model license -- read it before use (see README "Licenses").
#   bash setup/fetch_models.sh [base|ft|all]      (default: all)
set -euo pipefail
cd "$(dirname "$0")"; source ./versions.env; source ./env.sh
dl() { python3 -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" "$1" "$2" "$3"; }
python3 -c "import huggingface_hub" 2>/dev/null || { echo "pip install huggingface_hub first"; exit 1; }
what="${1:-all}"
[ "$what" = ft ]   || dl "$BASE_MODEL_REPO" "$BASE_MODEL_REV" "$GROOT_MODEL"
[ "$what" = base ] || dl "$FT_MODEL_REPO"   "$FT_MODEL_REV"   "$GROOT_FT_MODEL"
# expected sha256 of the weights we measured with (from the Hugging Face LFS index at those revisions)
python3 - "$GROOT_MODEL" "$GROOT_FT_MODEL" "$what" <<'PY'
import hashlib, os, sys
base, ft, what = sys.argv[1:4]
EXP = {base: {"model-00001-of-00003.safetensors": "4d5e8e2a81a4b25084965ae87b8257a87ebe2f3b6656f83bbb15ba13675940b6",
              "model-00002-of-00003.safetensors": "03ba0f11339d5ed24920582781c7352e308eb4eb78067c128bd8516b146656c8",
              "model-00003-of-00003.safetensors": "1f572eb204d7afe3ddbfb890ca56eac1a9bafbdce51ed6fd3ba314dc4298d565"},
       ft:   {"model-00001-of-00002.safetensors": "09ebd96da94f83244719020892fc7d5bce9de7bb56ccbac9789adf09e9b34b78",
              "model-00002-of-00002.safetensors": "51715a05c7d71b95727845c0def3a183fb43ee6a43f8abbf2284825c0d9ec7df"}}
dirs = {"all": [base, ft], "base": [base], "ft": [ft]}[what]
bad = 0
for d in dirs:
    for f, want in EXP[d].items():
        h = hashlib.sha256()
        with open(os.path.join(d, f), "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""): h.update(blk)
        ok = h.hexdigest() == want; bad += not ok
        print(("OK  " if ok else "BAD ") + os.path.join(d, f))
sys.exit(1 if bad else 0)
PY
