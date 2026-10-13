#!/usr/bin/env python3
"""Launch the M7 capture server (5090 FT policy) + run a short RoboCasa rollout to
capture real in-loop board raws for int8 calibration."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_ROBOCASA_PY = _gvrb_os.path.expanduser(_gvrb_os.environ.get("ROBOCASA_PY", _W+"/venv-robocasa/bin/python"))
_GVRB_ROOT = _gvrb_os.environ.get("GVRB_ROOT", _gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.realpath(__file__))))))
import os, signal, subprocess, sys, time

HERE = _W+"/closed_loop_m7"
N15 = _gvrb_os.environ.get("GROOT_REPO", _W+"/Isaac-GR00T")
STUBS = _gvrb_os.environ.get("GROOT_SHIM", _GVRB_ROOT+"/shims/x86")
M2 = _W+"/closed_loop_m2"
GROOT_PYBIN = _gvrb_os.environ.get("GROOT_PY", _W+"/venv-groot/bin/python")
ROBOVENV = _ROBOCASA_PY
ENV_NAME = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
FT = _gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain")
NOISE = _gvrb_os.environ.get("GVRB_NOISE_BANK", _W+"/noise_bank.npy")
PORT = 5573

def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    work = f"{HERE}/logs/cap_work"; os.makedirs(work, exist_ok=True)
    cap = f"{HERE}/calib_raws"
    env = dict(os.environ); env["PYTHONPATH"] = f"{STUBS}:{M2}:{N15}"
    slog = open(f"{work}/server.log", "w")
    srv = subprocess.Popen(
        [GROOT_PYBIN, f"{HERE}/capture_obs_server.py", "--model_path", FT, "--port", str(PORT),
         "--noise_bank", NOISE, "--denoising_steps", "4", "--work", work,
         "--capture_dir", cap, "--every", "3", "--max_frames", "48"],
        cwd=HERE, stdout=slog, stderr=subprocess.STDOUT, env=env)
    print("[CAP-RUN] server launching...", flush=True)
    ready, t0 = False, time.time()
    while time.time() - t0 < 400:
        if srv.poll() is not None:
            print("[CAP-RUN] server died; tail:", open(f"{work}/server.log").read()[-2000:]); sys.exit(1)
        if "SERVER READY" in open(f"{work}/server.log").read():
            ready = True; break
        time.sleep(3)
    if not ready:
        srv.send_signal(signal.SIGINT); sys.exit(1)
    print("[CAP-RUN] server READY", flush=True)
    cenv = dict(os.environ); cenv["PYTHONPATH"] = f"{STUBS}:{N15}"; cenv["MUJOCO_GL"] = "egl"
    clog = open(f"{work}/client.log", "w")
    rc = subprocess.call(
        [ROBOVENV, _W+"/closed_loop_m4/m4_rollout.py",
         "--env_name", ENV_NAME, "--n_episodes", str(n_ep), "--max_episode_steps", "720",
         "--base_seed", "0", "--port", str(PORT), "--out_dir", f"{work}/rollout"],
        stdout=clog, stderr=subprocess.STDOUT, env=cenv)
    print(f"[CAP-RUN] client rc={rc}", flush=True)
    srv.send_signal(signal.SIGINT)
    try: srv.wait(timeout=60)
    except Exception: srv.send_signal(signal.SIGINT); srv.wait(timeout=30)
    print("[CAP-RUN] done", flush=True)

if __name__ == "__main__":
    main()
