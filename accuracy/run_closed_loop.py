#!/usr/bin/env python3
"""Closed-loop accuracy run: RoboCasa GR1 PnPMilkToMicrowaveClose (pick up the milk, put it in the microwave, close the door), 20 seeds x 720 sim steps, one GR00T
policy server + the sim client on this x86 host (EGL render).

Server modes
  --server ref      reference_server.py on THIS host (RTX 5090, bf16)     -> the published 5090 reference
  --server local    policy_server.py on THIS host (--precision fp16/bf16/fp8)
  --server remote   policy_server.py on a Jetson over ssh (--ssh, --board_ip, --remote_*); the sim
                    client stays here and talks ZMQ to the board across the LAN

Every server gets the SAME noise bank (accuracy/gen_noise.py) and the client deep-reseeds every
scene, so different backends see bit-identical scenes + diffusion noise; only the arithmetic differs.
Servers are always stopped gracefully (SIGINT/SIGTERM -> the server dumps its call log). Never
SIGKILL a GPU process mid-run: on Jetson that can leave the GPU in a degraded state.

Environment (source setup/env.sh): GROOT_REPO, GROOT_FT_MODEL, GROOT_SHIM, GROOT_PY, ROBOCASA_PY.

Examples
  python accuracy/run_closed_loop.py --server ref --noise_bank noise_bank.npy --out_dir runs/r5090_bf16
  python accuracy/run_closed_loop.py --server remote --precision fp16 --ssh orin --board_ip 192.0.2.10 \\
      --remote_repo ~/groot-virtual-robot-bench --remote_isaac ~/Isaac-GR00T --remote_model ~/models/gr00t-ft \\
      --remote_python python3 --noise_bank noise_bank.npy --out_dir runs/orin_fp16
"""
import argparse, os, shlex, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"


def env(k):
    v = os.environ.get(k)
    if not v:
        sys.exit(f"{k} is not set -- run: source setup/env.sh (see docs/SETUP.md)")
    return v


def sh(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def wait_ready(read_log, alive, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        txt = read_log()
        if "SERVER READY" in txt:
            return True
        if "FP8-ABORT" in txt:
            print("[run] fp8 could not engage -- server aborted:\n" + txt[-3000:], flush=True)
            sys.exit(3)
        if not alive():
            print("[run] server died during startup:\n" + txt[-3000:], flush=True)
            sys.exit(1)
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", choices=["ref", "local", "remote"], required=True)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp8"], default="bf16")
    ap.add_argument("--fp8_backend", choices=["torch", "te"], default="torch")
    ap.add_argument("--noise_bank", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=5571)
    # remote (Jetson) server
    ap.add_argument("--ssh", help="ssh destination of the board (alias or user@host)")
    ap.add_argument("--board_ip", help="board address the sim client connects to")
    ap.add_argument("--remote_repo", help="this repo's path on the board")
    ap.add_argument("--remote_isaac", help="Isaac-GR00T (N1.5, sdpa-patched) path on the board")
    ap.add_argument("--remote_model", help="fine-tuned checkpoint path on the board")
    ap.add_argument("--remote_python", default="python3")
    a = ap.parse_args()

    out = os.path.abspath(a.out_dir); work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)
    noise = os.path.abspath(a.noise_bank)
    groot_repo, shim = env("GROOT_REPO"), env("GROOT_SHIM")

    if a.server in ("ref", "local"):
        script = "reference_server.py" if a.server == "ref" else "policy_server.py"
        cmd = [env("GROOT_PY"), os.path.join(HERE, script), "--model_path", env("GROOT_FT_MODEL"),
               "--port", str(a.port), "--noise_bank", noise, "--denoising_steps", "4", "--work", work]
        if a.server == "local":
            cmd += ["--precision", a.precision, "--fp8_backend", a.fp8_backend]
        senv = dict(os.environ, PYTHONPATH=f"{shim}:{groot_repo}", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        logp = os.path.join(work, "server.log")
        srv = subprocess.Popen(cmd, cwd=HERE, stdout=open(logp, "w"), stderr=subprocess.STDOUT, env=senv)
        ok = wait_ready(lambda: open(logp).read() if os.path.exists(logp) else "", lambda: srv.poll() is None)
        host = "localhost"

        def stop():
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(timeout=120)
            except subprocess.TimeoutExpired:
                srv.send_signal(signal.SIGINT); srv.wait(timeout=90)
    else:
        for k in ("ssh", "board_ip", "remote_repo", "remote_isaac", "remote_model"):
            if not getattr(a, k):
                sys.exit(f"--server remote needs --{k}")
        rwork = f"/tmp/gvrb_{a.precision}_{a.port}"
        q = shlex.quote
        sh(f"ssh {q(a.ssh)} mkdir -p {rwork}")
        r = sh(f"scp -q {q(noise)} {q(a.ssh)}:{rwork}/noise_bank.npy")
        if r.returncode:
            sys.exit("[run] could not copy the noise bank to the board: " + r.stderr)
        pp = f"{a.remote_repo}/shims/jetson:{a.remote_isaac}"   # the board is a Jetson: decord/flash_attn/pytorch3d import stubs
        # NOT "cd X && VAR=.. nohup ... &": that backgrounds the whole list, so $! is a wrapper subshell and stop()
        # would TERM the wrapper while the server keeps running (and keeps the port + GPU).
        srv_cmd = (f"cd {a.remote_repo}/accuracy || exit 1; PYTHONPATH={pp} CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                   f"nohup {a.remote_python} policy_server.py --model_path {a.remote_model} --precision {a.precision} "
                   f"--fp8_backend {a.fp8_backend} --port {a.port} --noise_bank {rwork}/noise_bank.npy "
                   f"--denoising_steps 4 --work {rwork} < /dev/null > {rwork}/server.log 2>&1 & echo $! > {rwork}/server.pid")
        r = sh(f"ssh -n {q(a.ssh)} {q(srv_cmd)}")   # -n + </dev/null: the backgrounded server must not hold the ssh channel open
        if r.returncode:
            sys.exit("[run] ssh launch failed: " + r.stderr)
        alive = lambda: sh(f"ssh {q(a.ssh)} 'kill -0 $(cat {rwork}/server.pid) 2>/dev/null && echo A'").stdout.strip() == "A"
        ok = wait_ready(lambda: sh(f"ssh {q(a.ssh)} cat {rwork}/server.log").stdout, alive)
        host = a.board_ip

        def stop():
            sh(f"ssh {q(a.ssh)} 'kill -TERM $(cat {rwork}/server.pid) 2>/dev/null'")
            for _ in range(12):
                time.sleep(5)
                if not alive():
                    break
                sh(f"ssh {q(a.ssh)} 'kill -TERM $(cat {rwork}/server.pid) 2>/dev/null'")
            for f in ("server_calllog.json", "fp8_proof.json", "server.log"):
                sh(f"scp -q {q(a.ssh)}:{rwork}/{f} {q(work)}/{f}")

    if not ok:
        print("[run] server not READY in time", flush=True); stop(); sys.exit(1)
    print(f"[run] server READY ({a.server}, {a.precision if a.server != 'ref' else 'bf16 reference'})", flush=True)

    cenv = dict(os.environ, PYTHONPATH=f"{shim}:{groot_repo}", MUJOCO_GL="egl")
    rc = subprocess.call([env("ROBOCASA_PY"), os.path.join(HERE, "rollout_client.py"),
                          "--env_name", ENV_NAME, "--n_episodes", str(a.n_episodes),
                          "--max_episode_steps", str(a.max_episode_steps), "--base_seed", str(a.base_seed),
                          "--host", host, "--port", str(a.port), "--out_dir", out],
                         stdout=open(os.path.join(work, "client.log"), "w"), stderr=subprocess.STDOUT, env=cenv)
    print(f"[run] client exited rc={rc}; stopping server gracefully", flush=True)
    stop()
    print(f"[run] results -> {out} (rollout_summary.json, episode_*_actions.npy, work/)", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
