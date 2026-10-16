# source setup/env.sh   -- every script reads its paths from these variables. Override any of them
# before sourcing (or edit here). GVRB_WORK holds everything large: upstream clones, models, ONNX, engines.
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export GVRB_ROOT="${GVRB_ROOT:-$_here}"
export GVRB_WORK="${GVRB_WORK:-$HOME/gvrb}"
export GROOT_REPO="${GROOT_REPO:-$GVRB_WORK/Isaac-GR00T}"                 # Isaac-GR00T @ N1.5 commit (+ patch)
export GROOT_MODEL="${GROOT_MODEL:-$GVRB_WORK/models/GR00T-N1.5-3B}"       # base model: speed benchmarks
export GROOT_FT_MODEL="${GROOT_FT_MODEL:-$GVRB_WORK/models/gr00t-n1.5-robocasa-tabletop-posttrain}"  # closed loop
export GROOT_ONNX="${GROOT_ONNX:-$GVRB_WORK/onnx}"                         # fp16 ONNX from speed/export_onnx.sh
export GROOT_STEP_NPZ="${GROOT_STEP_NPZ:-$GVRB_WORK/groot_step0.npz}"      # speed input (speed/make_step0.py)
if [ "$(uname -m)" = "aarch64" ]; then
  export GROOT_SHIM="${GROOT_SHIM:-$GVRB_ROOT/shims/jetson}"; export JETSON="${JETSON:-1}"
else
  export GROOT_SHIM="${GROOT_SHIM:-$GVRB_ROOT/shims/x86}"
fi
if [ -z "${GROOT_PY:-}" ] && [ "$(uname -m)" = aarch64 ] && [ ! -x "$GVRB_WORK/venv-groot/bin/python" ]; then
  GROOT_PY=python3                                                         # Orin: system python3 (setup/setup_orin_jp6.sh)
fi
export GROOT_PY="${GROOT_PY:-$GVRB_WORK/venv-groot/bin/python}"            # policy / TensorRT env
export ROBOCASA_PY="${ROBOCASA_PY:-$GVRB_WORK/venv-robocasa/bin/python}"   # simulator env (x86 only)
export PYTHONPATH="$GROOT_SHIM:$GROOT_REPO${PYTHONPATH:+:$PYTHONPATH}"
# --- optional: IQ-9075 (Qualcomm) + Jetson profiling (see iq9/README.md, profiling/README.md) ---
export QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/opt/qcom/qairt}"                  # x86 QAIRT/QNN SDK root (contains 2.47.0.260601/)
export IQ9_WORK="${IQ9_WORK:-/opt/gvrb}"                                   # work dir ON the IQ-9075
export IQ9_QNN_ROOT="${IQ9_QNN_ROOT:-/opt/qnn}"                            # QNN runtime ON the IQ-9075
export IQ9_PY_ROOT="${IQ9_PY_ROOT:-/opt/iq9-python}"                       # Python tree ON the IQ-9075
export GVRB_BOARD_WORK="${GVRB_BOARD_WORK:-$HOME/gvrb-board}"              # engine/profile dir ON a Jetson (profiling/)
unset _here
# drone benchmark paths: DRONE_WORK, DRONE_ENG, DRONE_ASSETS, DRONE_STANDINS, DRONE_RUNS (drone/README.md)
[ -f "$GVRB_ROOT/drone/env.sh" ] && source "$GVRB_ROOT/drone/env.sh"
