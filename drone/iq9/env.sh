# source setup/env.sh; source drone/env.sh; source drone/iq9/env.sh   -- paths for the drone policy on the IQ-9075.
# Override any of them before sourcing. Host = the x86 machine with the QAIRT SDK; board = the IQ-9075.
# setup/env.sh already provides QAIRT_SDK_ROOT (x86 SDK root, contains $QAIRT_VER/), IQ9_WORK (board dir of the GR00T port:
# optionB/ctx_fix, ctx_aaf0, glue_constants, io), IQ9_QNN_ROOT (board QNN runtime: bin/, lib/, skel/ = the v73 skel libs)
# and IQ9_PY_ROOT (board Python tree).
export QAIRT_VER="${QAIRT_VER:-2.47.0.260601}"
export IQ9_HOST="${IQ9_HOST:-<board>}"                                          # ssh target of the board (key auth, root)
export IQ9_CXX="${IQ9_CXX:-aarch64-linux-gnu-g++-11}"
# --- host ---
export DRONE_IQ9_WORK="${DRONE_IQ9_WORK:-$DRONE_WORK/iq9}"                       # host: builds, gate frames, staged board tree
# --- board (stage_board.sh copies everything here and writes $DRONE_IQ9_BOARD_DIR/env.sh with these values) ---
export DRONE_IQ9_BOARD_DIR="${DRONE_IQ9_BOARD_DIR:-/opt/gvrb-drone}"           # board: scripts, bins, frames, runs
export DRONE_IQ9_PY="${DRONE_IQ9_PY:-$IQ9_PY_ROOT/bin/python3}"                  # board: python with numpy, torch (CPU), GR00T's deps
export DRONE_IQ9_BLAS="${DRONE_IQ9_BLAS:-$IQ9_PY_ROOT/lib/python3.12/site-packages/torch/lib/libopenblas.so.0}"  # board: any LP64 libopenblas
export DRONE_IQ9_GLUE="${DRONE_IQ9_GLUE:-$IQ9_WORK/optionB/glue_constants}"    # board: mlp1 / embedding constants (iq9/README.md step 1.5)
export DRONE_IQ9_CONST="${DRONE_IQ9_CONST:-$IQ9_WORK/optionB/io}"              # board: dir holding position_ids.raw (groot_capture_ref.py)
export DRONE_IQ9_VIT_FIX="${DRONE_IQ9_VIT_FIX:-$IQ9_WORK/optionB/ctx_fix/groot_vit_fp16.bin}"
export DRONE_IQ9_VIT_AAF0="${DRONE_IQ9_VIT_AAF0:-$IQ9_WORK/optionB/ctx_aaf0/groot_vit_aaf0_fp16.bin}"
export DRONE_IQ9_LLM="${DRONE_IQ9_LLM:-$IQ9_WORK/optionB/ctx_fix/groot_llm_fp16.bin}"
