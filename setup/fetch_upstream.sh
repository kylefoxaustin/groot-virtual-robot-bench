#!/usr/bin/env bash
# Clone the pinned upstreams into $GVRB_WORK and apply our one patch to Isaac-GR00T.
#   bash setup/fetch_upstream.sh            # Isaac-GR00T only (enough on a Jetson)
#   bash setup/fetch_upstream.sh --sim      # + robosuite + robocasa-gr1-tabletop-tasks (x86 sim host)
set -euo pipefail
cd "$(dirname "$0")"; source ./versions.env; source ./env.sh
mkdir -p "$GVRB_WORK"
# No git-lfs binary but an lfs filter in your git config would break the checkout: switch the filter off for these clones
# (nothing the benchmarks use is in LFS).
NOLFS=(); command -v git-lfs >/dev/null || NOLFS=(-c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.clean= -c filter.lfs.required=false)
clone() {  # url commit dir
  if [ ! -d "$3/.git" ]; then GIT_LFS_SKIP_SMUDGE=1 git "${NOLFS[@]}" clone -q "$1" "$3"; fi
  (cd "$3" && GIT_LFS_SKIP_SMUDGE=1 git "${NOLFS[@]}" -c advice.detachedHead=false checkout -q "$2")
  echo "$(basename "$3") @ $(cd "$3" && git rev-parse --short HEAD)"
}
clone "$ISAAC_GR00T_URL" "$ISAAC_GR00T_COMMIT" "$GROOT_REPO"
PATCH="$GVRB_ROOT/setup/patches/isaac-gr00t-n15-export_onnx.patch"
if (cd "$GROOT_REPO" && git apply --check "$PATCH" 2>/dev/null); then
  (cd "$GROOT_REPO" && git apply "$PATCH") && echo "patched export_onnx.py"
elif (cd "$GROOT_REPO" && git apply --reverse --check "$PATCH" 2>/dev/null); then
  echo "export_onnx.py patch already applied -- skipping"
else
  echo "ERROR: $PATCH does not apply to $GROOT_REPO (local edits, or not at commit $ISAAC_GR00T_COMMIT)." >&2
  echo "       Inspect with: cd $GROOT_REPO && git status && git apply --check -v $PATCH" >&2
  exit 1
fi
if [ "${1:-}" = "--sim" ]; then
  clone "$ROBOSUITE_URL" "$ROBOSUITE_COMMIT" "$GVRB_WORK/robosuite"
  clone "$ROBOCASA_GR1_URL" "$ROBOCASA_GR1_COMMIT" "$GVRB_WORK/robocasa-gr1-tabletop-tasks"
fi
