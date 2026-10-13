#!/usr/bin/env python3
"""closed_loop_m7 closed-loop runner: launch the iq9 server with a chosen ctx-bin set (fp16 /
armA int8-body+fp16-head / armB all-int8) and run the 20-seed RoboCasa rollout with the
shared noise bank ($GVRB_NOISE_BANK, default $GVRB_WORK/noise_bank.npy), bit-matched to the 5090 reference. Clean SIGINT shutdown (never SIGKILL)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_ROBOCASA_PY = _gvrb_os.path.expanduser(_gvrb_os.environ.get("ROBOCASA_PY", _W+"/venv-robocasa/bin/python"))
_GVRB_ROOT = _gvrb_os.environ.get("GVRB_ROOT", _gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.dirname(_gvrb_os.path.realpath(__file__))))))
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
import argparse, os, signal, subprocess, sys, time
HERE=_W+"/closed_loop_m7"; N15=_gvrb_os.environ.get("GROOT_REPO", _W+"/Isaac-GR00T")
STUBS=_gvrb_os.environ.get("GROOT_SHIM", _GVRB_ROOT+"/shims/x86"); M2=_W+"/closed_loop_m2"
GROOT_PYBIN=_gvrb_os.environ.get("GROOT_PY", _W+"/venv-groot/bin/python"); ROBOVENV=_ROBOCASA_PY
ENV_NAME="gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
FT=_gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain"); NOISE=_gvrb_os.environ.get("GVRB_NOISE_BANK", _W+"/noise_bank.npy")
BROOT=_B+"/optionB"; I8=f"{BROOT}/ctx_int8"; FP=f"{BROOT}/ctx_ft"

def ctx_list(arm):
    fp={n:f"{FP}/{n}_fp16.bin" for n in ["groot_vit","groot_llm","vlln_vl_self_attention",
        "state_encoder","action_encoder","groot_dit","action_decoder"]}
    # BODY uses percentile-99.9 int8 (best-practice, outlier-robust); HEAD uses min-max int8
    bodyp={n:f"{I8}/{n}_int8p.bin" for n in ["groot_vit","groot_llm","vlln_vl_self_attention_47"]}
    i8={n:f"{I8}/{n}_int8.bin" for n in ["state_encoder_47","action_encoder_47","groot_dit","action_decoder_47"]}
    BODY_I8=[bodyp["groot_vit"],bodyp["groot_llm"],bodyp["vlln_vl_self_attention_47"]]
    BODY_FP=[fp["groot_vit"],fp["groot_llm"],fp["vlln_vl_self_attention"]]
    HEAD_I8=[i8["state_encoder_47"],i8["action_encoder_47"],i8["groot_dit"],i8["action_decoder_47"]]
    HEAD_FP=[fp["state_encoder"],fp["action_encoder"],fp["groot_dit"],fp["action_decoder"]]
    headp={n:f"{I8}/{n}_int8p.bin" for n in ["state_encoder_47","action_encoder_47","groot_dit","action_decoder_47"]}
    HEAD_I8P=[headp["state_encoder_47"],headp["action_encoder_47"],headp["groot_dit"],headp["action_decoder_47"]]
    # W8A16 (int8 per-channel weights + int16 activations) — transformer-safe recipe
    a16b={n:f"{I8}/{n}_a16.bin" for n in ["groot_vit","groot_llm","vlln_vl_self_attention_47"]}
    a16h={n:f"{I8}/{n}_a16.bin" for n in ["state_encoder_47","action_encoder_47","groot_dit","action_decoder_47"]}
    BODY_A16=[a16b["groot_vit"],a16b["groot_llm"],a16b["vlln_vl_self_attention_47"]]
    HEAD_A16=[a16h["state_encoder_47"],a16h["action_encoder_47"],a16h["groot_dit"],a16h["action_decoder_47"]]
    # W8A16 + per-ROW (per-output-channel) weights — the CORRECT transformer int8 recipe
    prb={n:f"{I8}/{n}_a16pr.bin" for n in ["groot_vit","groot_llm","vlln_vl_self_attention_47"]}
    BODY_PR=[prb["groot_vit"],prb["groot_llm"],prb["vlln_vl_self_attention_47"]]
    prh={n:f"{I8}/{n}_a16pr.bin" for n in ["state_encoder_47","action_encoder_47","groot_dit","action_decoder_47"]}
    HEAD_PR=[prh["state_encoder_47"],prh["action_encoder_47"],prh["groot_dit"],prh["action_decoder_47"]]
    return {"fp16":BODY_FP+HEAD_FP, "armA":BODY_I8+HEAD_FP,
            "armB":BODY_I8+HEAD_I8, "armC":BODY_FP+HEAD_I8,
            "armCp":BODY_FP+HEAD_I8P, "armBp":BODY_I8+HEAD_I8P,
            "a16all":BODY_A16+HEAD_A16, "a16body":BODY_A16+HEAD_FP,
            "a16head":BODY_FP+HEAD_A16,
            "prbody":BODY_PR+HEAD_FP, "prall":BODY_PR+HEAD_A16,
            "prhead":BODY_FP+HEAD_PR}[arm]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=["fp16","armA","armB","armC","armCp","armBp","a16all","a16body","a16head","prbody","prall","prhead"],required=True)
    ap.add_argument("--n_episodes",type=int,default=20); ap.add_argument("--base_seed",type=int,default=0)
    ap.add_argument("--max_episode_steps",type=int,default=720); ap.add_argument("--port",type=int,default=5578)
    ap.add_argument("--work",required=True); ap.add_argument("--out_dir",required=True)
    args=ap.parse_args(); os.makedirs(args.work,exist_ok=True); os.makedirs(args.out_dir,exist_ok=True)
    env=dict(os.environ); env["PYTHONPATH"]=f"{STUBS}:{M2}:{N15}"
    env["M7_CTX_LIST"]=",".join(ctx_list(args.arm))
    print(f"[M7-RUN] arm={args.arm} ctx set:\n  "+"\n  ".join(ctx_list(args.arm)),flush=True)
    slog=open(os.path.join(args.work,"server.log"),"w")
    srv=subprocess.Popen([GROOT_PYBIN,os.path.join(HERE,"m7_server_iq9.py"),"--model_path",FT,
        "--port",str(args.port),"--noise_bank",NOISE,"--work",args.work],
        cwd=HERE,stdout=slog,stderr=subprocess.STDOUT,env=env)
    ready,t0=False,time.time()
    while time.time()-t0<420:
        if srv.poll() is not None:
            print("[M7-RUN] server died:\n",open(os.path.join(args.work,"server.log")).read()[-3000:]);sys.exit(1)
        if "SERVER READY" in open(os.path.join(args.work,"server.log")).read(): ready=True;break
        time.sleep(3)
    if not ready:
        print("[M7-RUN] not ready");srv.send_signal(signal.SIGINT);srv.wait(timeout=90);sys.exit(1)
    print("[M7-RUN] server READY",flush=True)
    cenv=dict(os.environ); cenv["PYTHONPATH"]=f"{STUBS}:{N15}"; cenv["MUJOCO_GL"]="egl"
    clog=open(os.path.join(args.work,"client.log"),"w")
    rc=subprocess.call([ROBOVENV,_W+"/closed_loop_m4/m4_rollout.py",
        "--env_name",ENV_NAME,"--n_episodes",str(args.n_episodes),
        "--max_episode_steps",str(args.max_episode_steps),"--base_seed",str(args.base_seed),
        "--port",str(args.port),"--out_dir",args.out_dir],stdout=clog,stderr=subprocess.STDOUT,env=cenv)
    print(f"[M7-RUN] client rc={rc}",flush=True)
    print("[M7-RUN] SIGINT server (clean; never SIGKILL)...",flush=True)
    srv.send_signal(signal.SIGINT)
    try: srv.wait(timeout=150)
    except subprocess.TimeoutExpired:
        srv.send_signal(signal.SIGINT); srv.wait(timeout=90)
    print("[M7-RUN] server exited rc=",srv.returncode,flush=True); sys.exit(rc)

if __name__=="__main__": main()
