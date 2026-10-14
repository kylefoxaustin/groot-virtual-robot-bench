# source sonic/env.sh   -- paths for the GEAR-SONIC benchmark (sonic/). Override any of them before sourcing.
# SONIC_WORK holds everything large: the upstream checkout, models, TensorRT engines, run directories.
export SONIC_COMMIT="${SONIC_COMMIT:-b042411fae38ee4d1af9aac82a37a1f8d14d6dd0}"      # NVlabs/GR00T-WholeBodyControl, 2026-09-23
export SONIC_WORK="${SONIC_WORK:-${GVRB_WORK:-$HOME/gvrb}/sonic}"
export SONIC_REPO="${SONIC_REPO:-$SONIC_WORK/GR00T-WholeBodyControl}"                # upstream checkout (+ sonic/harness/sonic_probe.patch)
export SONIC_DEPLOY="${SONIC_DEPLOY:-$SONIC_REPO/gear_sonic_deploy}"
export SONIC_SIM_PY="${SONIC_SIM_PY:-$SONIC_WORK/venv-sim/bin/python}"               # python with mujoco 3.15, unitree_sdk2py, cyclonedds
export SONIC_RUNS="${SONIC_RUNS:-$SONIC_WORK/runs}"
# x86 only: the TensorRT tarball upstream requires (10.13.0.35), bind-mounted into the deploy container
export SONIC_TRT="${SONIC_TRT:-$SONIC_WORK/TensorRT-10.13.0.35}"
# Jetson only: onnxruntime 1.16.3 aarch64 release (upstream CMake links it; same version as the x86 container)
export SONIC_ORT="${SONIC_ORT:-$SONIC_WORK/onnxruntime-linux-aarch64-1.16.3}"
