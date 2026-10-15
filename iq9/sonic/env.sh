# source setup/env.sh; source sonic/env.sh; source iq9/sonic/env.sh   -- paths for the GEAR-SONIC port to the IQ-9075.
# Override any of them before sourcing. Host = the x86 machine with the QAIRT SDK; board = the IQ-9075.
# setup/env.sh already provides QAIRT_SDK_ROOT (x86 SDK root, contains $QAIRT_VER/), IQ9_QNN_ROOT (board QNN runtime with
# bin/, lib/ and skel/ = the v73 skel libs) and IQ9_PY_ROOT (board Python tree, bin/python3).
export QAIRT_VER="${QAIRT_VER:-2.47.0.260601}"
export IQ9_HOST="${IQ9_HOST:-<board>}"                                        # ssh target of the board (key auth, root)
export IQ9_SONIC_WORK="${IQ9_SONIC_WORK:-$SONIC_WORK/iq9}"                   # host: graphs, DLCs, context bins, builds, staged board tree
export IQ9_SONIC_BOARD_DIR="${IQ9_SONIC_BOARD_DIR:-/opt/sonic}"              # board: where stage_board.sh puts everything
export IQ9_SONIC_SAMPLEAPP="${IQ9_SONIC_SAMPLEAPP:-$IQ9_SONIC_WORK/qnn_sampleapp_src}"  # host: copy of the SDK's SampleApp src (see README)
export IQ9_SONIC_ORT="${IQ9_SONIC_ORT:-$SONIC_WORK/onnxruntime-linux-aarch64-1.20.1}"  # host: onnxruntime 1.20.1 aarch64 release (linked + shipped)
export IQ9_SONIC_SYSROOT="${IQ9_SONIC_SYSROOT:-$IQ9_SONIC_WORK/sysroot}"     # host: aarch64 headers/libs the deploy links (build_sysroot.sh)
export IQ9_PYVER="${IQ9_PYVER:-3.12}"                                         # the board Python's version (the cyclonedds _clayer is built for it)
export IQ9_CXX="${IQ9_CXX:-aarch64-linux-gnu-g++-11}"
export IQ9_CC="${IQ9_CC:-aarch64-linux-gnu-gcc-11}"
