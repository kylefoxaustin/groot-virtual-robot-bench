#!/usr/bin/env python3
"""Fine-tuned closed-loop driver (closed_loop_m3's orchestrator, pointed at the FINE-TUNED
checkpoint $GROOT_FT_MODEL): launch a backend server (5090 reference or iq9), run the
RoboCasa rollout client (m4_rollout.py) over seeds base_seed..+n_episodes-1 with an INJECTED
fixed noise bank, then cleanly shut the server (SIGINT -> board QUIT for iq9; never SIGKILL).

  --backend 5090  : closed_loop_m3/m3_server_5090.py (Gr00tPolicy bf16 cuda, denoise=4, prefix-stripped)
  --backend iq9   : m4_server_iq9.py   (Hexagon v73 fp16 GFRAME loop on the fine-tuned bins:
                    $IQ9_WORK/optionB/{ctx_ft,glue_ft,io_ft}, deployed by build/deploy.sh)
  --zero_image    : iq9 only; zero the pixel_values (in-loop obs-null arm)
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_ROBOCASA_PY = _gvrb_os.path.expanduser(_gvrb_os.environ.get("ROBOCASA_PY", _W+"/venv-robocasa/bin/python"))
_GVRB_ROOT = _gvrb_os.environ.get("GVRB_ROOT", _gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.realpath(__file__))))))
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
N15 = _gvrb_os.environ.get("GROOT_REPO", _W+"/Isaac-GR00T")
STUBS = _gvrb_os.environ.get("GROOT_SHIM", _GVRB_ROOT+"/shims/x86")
M2 = _W+"/closed_loop_m2"
GROOT_PYBIN = _gvrb_os.environ.get("GROOT_PY", _W+"/venv-groot/bin/python")
ROBOVENV = _ROBOCASA_PY
ENV_NAME = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def wait_for_gpu(min_mb, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = gpu_free_mb()
        print(f"[M3-RUN] gpu free {f} MB (need >= {min_mb})", flush=True)
        if f < 0 or f >= min_mb:
            return f
        time.sleep(20)
    return gpu_free_mb()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["5090", "iq9"], required=True)
    ap.add_argument("--noise_bank", required=True)
    ap.add_argument("--zero_image", action="store_true")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=5571)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{STUBS}:{M2}:{N15}"

    FT = _gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain")   # fine-tuned checkpoint
    M3 = _W+"/closed_loop_m3"
    if args.backend == "5090":
        srv_cmd = [GROOT_PYBIN, os.path.join(M3, "m3_server_5090.py"),
                   "--model_path", FT,
                   "--port", str(args.port), "--noise_bank", args.noise_bank,
                   "--denoising_steps", "4", "--work", args.work]
        # 5090 arm needs the GPU for BOTH the policy (~1.5GB) and the EGL render
        min_gpu = 4000
    else:
        srv_cmd = [GROOT_PYBIN, os.path.join(HERE, "m4_server_iq9.py"),
                   "--model_path", FT,
                   "--port", str(args.port), "--noise_bank", args.noise_bank,
                   "--work", args.work]
        if args.zero_image:
            srv_cmd.append("--zero_image")
        min_gpu = 900  # iq9 policy on-board; GPU only for EGL render

    slog = open(os.path.join(args.work, "server.log"), "w")
    srv = subprocess.Popen(srv_cmd, cwd=HERE, stdout=slog, stderr=subprocess.STDOUT, env=env)
    print(f"[M3-RUN] {args.backend} server launching...", flush=True)
    ready, t0 = False, time.time()
    while time.time() - t0 < 300:
        if srv.poll() is not None:
            print("[M3-RUN] server died during startup; see server.log", flush=True)
            sys.exit(1)
        try:
            with open(os.path.join(args.work, "server.log")) as f:
                if "SERVER READY" in f.read():
                    ready = True; break
        except FileNotFoundError:
            pass
        time.sleep(2)
    if not ready:
        print("[M3-RUN] server not ready in time", flush=True)
        srv.send_signal(signal.SIGINT); srv.wait(timeout=90); sys.exit(1)
    print("[M3-RUN] server READY", flush=True)

    wait_for_gpu(min_gpu)

    cenv = dict(os.environ)
    cenv["PYTHONPATH"] = f"{STUBS}:{N15}"
    cenv["MUJOCO_GL"] = "egl"
    clog = open(os.path.join(args.work, "client.log"), "w")
    rc = subprocess.call(
        [ROBOVENV, os.path.join(HERE, "m4_rollout.py"),
         "--env_name", ENV_NAME, "--n_episodes", str(args.n_episodes),
         "--max_episode_steps", str(args.max_episode_steps), "--base_seed", str(args.base_seed),
         "--port", str(args.port), "--out_dir", args.out_dir],
        stdout=clog, stderr=subprocess.STDOUT, env=cenv)
    print(f"[M3-RUN] client exited rc={rc}", flush=True)

    print("[M3-RUN] SIGINT server for clean shutdown...", flush=True)
    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("[M3-RUN] server slow; SIGINT again", flush=True)
        srv.send_signal(signal.SIGINT); srv.wait(timeout=90)
    print("[M3-RUN] server exited rc=", srv.returncode, flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
