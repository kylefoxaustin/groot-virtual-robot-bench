#!/usr/bin/env python3
"""RESIDENT GR00T N1.5-3B control loop on iq9 — AH-FIX build.
Same pipeline as resident_loop_groot.py but with the fixed action-head bins (ctx_fix):
  - state_encoder_47 / action_decoder_47: embodiment baked in -> NO embodiment_id input
  - action_encoder_47: embodiment baked + sinusoidal removed -> inputs = (actions, time_proj)
    where time_proj[t] = time_emb[t] @ W2b[eid] is precomputed on host (io/time_proj_t{td}.raw)
Heavy graphs (vit/llm/vlln/dit) unchanged. Adds --null-image (zero pixel_values) for the null arm.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, sys, time, subprocess, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host_glue_groot import GrootGlue

BROOT = os.environ.get("GROOT_BROOT", _B+"/optionB")
CTX = f"{BROOT}/ctx_fix"; IO = f"{BROOT}/io"; GLUE = f"{BROOT}/glue_constants"
QNN = os.environ.get("QNN_ROOT", _BQ+"")

CTX_BINS = {
    f"{CTX}/groot_vit_fp16.bin": "groot_vit",
    f"{CTX}/groot_llm_fp16.bin": "groot_llm",
    f"{CTX}/vlln_vl_self_attention_fp16.bin": "vlln_vl_self_attention_47",
    f"{CTX}/state_encoder_fp16.bin": "state_encoder_47",
    f"{CTX}/action_encoder_fp16.bin": "action_encoder_47",
    f"{CTX}/groot_dit_fp16.bin": "groot_dit",
    f"{CTX}/action_decoder_fp16.bin": "action_decoder_47",
}
GN = {"vit":"groot_vit", "llm":"groot_llm", "vlln":"vlln_vl_self_attention_47",
      "state_encoder":"state_encoder_47", "action_encoder":"action_encoder_47",
      "DiT":"groot_dit", "action_decoder":"action_decoder_47"}
PROTO = ("LOADED", "READY", "OK", "PONG", "BYE", "ERR")
IMAGE_TOKEN_INDEX = int(os.environ.get("GROOT_IMAGE_TOKEN", "151669"))


class Resident:
    def __init__(self, outdir, log_level="error"):
        self.outdir = outdir; self._dirs = set(); os.makedirs(outdir, exist_ok=True)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{QNN}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["ADSP_LIBRARY_PATH"] = f"{QNN}/skel"
        here = os.path.dirname(os.path.abspath(__file__))
        args = [f"{here}/resident_orch", "--backend", f"{QNN}/lib/libQnnHtp.so",
                "--system_library", f"{QNN}/lib/libQnnSystem.so", "--log_level", log_level]
        for path in CTX_BINS:
            args += ["--ctx", path]
        self.errlog = open(f"{outdir}/orch_stderr.log", "w")
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errlog, env=env, text=True, bufsize=1)
        self.loaded = []; self.load_us = None
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("resident_orch died during load; see orch_stderr.log")
            line = line.strip()
            if line.startswith("LOADED"): self.loaded.append(line.split()[1])
            elif line.startswith("READY"):
                for tok in line.split():
                    if tok.startswith("load_us="): self.load_us = int(tok.split("=")[1])
                break
            elif line.startswith("ERR"): raise RuntimeError(f"load error: {line}")

    def _readline_proto(self):
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("resident_orch closed stdout")
            line = line.strip()
            if line.startswith(PROTO): return line

    def exec_graph(self, name, inputs, tag):
        il = f"{self.outdir}/il_{tag}.txt"
        with open(il, "w") as f:
            f.write(" ".join(f"{k}:={v}" for k, v in inputs.items()) + "\n")
        odir = f"{self.outdir}/{tag}"
        if tag not in self._dirs:
            os.makedirs(f"{odir}/Result_0", exist_ok=True); self._dirs.add(tag)
        self.p.stdin.write(f"EXEC {name} {il} {odir}\n"); self.p.stdin.flush()
        resp = self._readline_proto()
        if not resp.startswith("OK"): raise RuntimeError(f"exec {name} failed: {resp}")
        exec_us = int(resp.split()[2])
        outs = {os.path.basename(f)[:-4]: np.fromfile(f, np.float32)
                for f in glob.glob(f"{odir}/Result_0/*.raw")}
        return outs, exec_us

    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(timeout=30)
        except Exception:
            self.p.terminate()
        self.errlog.close()


def w(path, arr, dt=np.float32):
    arr.astype(dt).tofile(path); return path


def run_frame(R, g, timers, pixel_path):
    def rec(k, e): timers.setdefault(k, []).append(e)
    o, e = R.exec_graph(GN["vit"], {"pixel_values": pixel_path,
                                "position_ids": f"{IO}/position_ids.raw"}, "vit"); rec("vit", e)
    vit_embeds = list(o.values())[0].reshape(1, 256, 1152)
    proj = g.mlp1_project(vit_embeds)
    input_ids = np.fromfile(f"{IO}/input_ids.raw", np.int64).reshape(1, -1)
    inputs_embeds = g.embed_and_scatter(input_ids, proj, IMAGE_TOKEN_INDEX)
    N = inputs_embeds.shape[1]
    w(f"{IO}/_inputs_embeds.raw", inputs_embeds)
    o, e = R.exec_graph(GN["llm"], {"inputs_embeds": f"{IO}/_inputs_embeds.raw",
                                "attention_mask": f"{IO}/attention_mask.raw"}, "llm"); rec("llm", e)
    backbone = list(o.values())[0].reshape(1, N, 2048)
    w(f"{IO}/_backbone.raw", backbone)
    o, e = R.exec_graph(GN["vlln"], {"backbone_features": f"{IO}/_backbone.raw"}, "vlln"); rec("vlln", e)
    vl_embs = list(o.values())[0].reshape(1, -1, 2048)
    w(f"{IO}/_vl_embs.raw", vl_embs)
    # state_encoder: embodiment BAKED -> state only
    o, e = R.exec_graph(GN["state_encoder"], {"state": f"{IO}/state.raw"}, "state"); rec("state", e)
    state_features = list(o.values())[0].reshape(1, 1, 1536)
    actions = np.fromfile(f"{IO}/init_actions.raw", np.float32).reshape(1, 16, 32)
    dit_e = []
    for s in range(g.num_steps):
        td = g.t_discretized(s)                                     # 0,250,500,750
        w(f"{IO}/_actions.raw", actions)
        w(f"{IO}/_ts.raw", np.array([td], np.float32), np.float32)  # DiT still needs the int timestep
        # action_encoder: embodiment baked + sinusoidal precomputed -> (actions, time_proj)
        o, e = R.exec_graph(GN["action_encoder"], {"actions": f"{IO}/_actions.raw",
                             "time_proj": f"{IO}/time_proj_t{td}.raw"}, "aenc"); rec("aenc", e)
        af = g.add_action_pos_embed(list(o.values())[0].reshape(1, 16, 1536))
        sa = g.assemble_sa_embs(state_features, af)
        w(f"{IO}/_sa.raw", sa)
        o, e = R.exec_graph(GN["DiT"], {"sa_embs": f"{IO}/_sa.raw", "vl_embs": f"{IO}/_vl_embs.raw",
                                    "timesteps_tensor": f"{IO}/_ts.raw"}, "dit"); dit_e.append(e)
        mo = list(o.values())[0].reshape(1, 49, 1024)
        w(f"{IO}/_mo.raw", mo)
        # action_decoder: embodiment BAKED -> model_output only
        o, e = R.exec_graph(GN["action_decoder"], {"model_output": f"{IO}/_mo.raw"}, "adec"); rec("adec", e)
        pred = list(o.values())[0].reshape(1, 49, 32)
        actions = g.euler(actions, pred)
    timers.setdefault("dit_step", []).extend(dit_e)
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--outdir", default=f"{BROOT}/out/resident_fix")
    ap.add_argument("--golden", default=f"{IO}/groot_fresh_golden.npy")
    ap.add_argument("--null-image", action="store_true", help="zero pixel_values (null arm)")
    args = ap.parse_args()
    g = GrootGlue(GLUE)
    pixel_path = f"{IO}/pixel_values.raw"
    if args.null_image:
        pv = np.zeros_like(np.fromfile(f"{IO}/pixel_values.raw", np.float32))
        pixel_path = f"{IO}/_pixel_null.raw"; w(pixel_path, pv)
    R = Resident(args.outdir)
    print(f"[groot-fix] loaded {len(R.loaded)} graphs: {R.loaded}; ctx load {R.load_us/1e3:.1f} ms", flush=True)
    if len(R.loaded) != 7: print(f"WARNING expected 7 graphs, got {len(R.loaded)}", flush=True)
    walls, timers, last = [], {}, None
    try:
        for fr in range(args.frames):
            t0 = time.perf_counter(); last = run_frame(R, g, timers, pixel_path); wdt = time.perf_counter() - t0
            walls.append(wdt)
            print(f"[frame {fr}] wall={wdt*1000:.1f} ms mean|a|={np.abs(last).mean():.5f}", flush=True)
    finally:
        if last is not None:
            np.save(f"{args.outdir}/actions_groot_resident_fix.npy", last)
        R.close()
    fw = np.array(walls); steady = fw[args.warmup:]
    med = lambda k: (float(np.median(timers[k]))/1e3 if timers.get(k) else None)
    out = {"date": time.strftime("%Y-%m-%d"), "tag": "MEASURED", "frames": args.frames, "null_image": args.null_image,
           "steady_median_ms": round(float(np.median(steady))*1000, 2),
           "control_hz": round(1.0/float(np.median(steady)), 3),
           "per_graph_ms": {k: med(k) for k in ["vit","llm","vlln","state","aenc","dit_step","adec"]},
           "one_time_ctx_load_ms": round(R.load_us/1e3, 1)}
    if os.path.exists(args.golden):
        gold = np.load(args.golden).astype(np.float64).reshape(1, 16, 32)
        a = last.astype(np.float64)
        rel = float(np.mean(np.abs(a-gold))/(np.mean(np.abs(gold))+1e-12))
        cos = float(np.dot(a.ravel(), gold.ravel())/(np.linalg.norm(a)*np.linalg.norm(gold)+1e-12))
        rreal = float(np.mean(np.abs(a[...,:26]-gold[...,:26]))/(np.mean(np.abs(gold[...,:26]))+1e-12))
        out["action_gate_vs_golden"] = {"rel_mae_32d": round(rel,5), "rel_mae_26real": round(rreal,5),
            "cos": round(cos,6), "fp16_floor_5e-3": rel < 5e-3, "bf16_gate_0.04": rel < 0.04}
    np.save(f"{args.outdir}/action_final.npy", last)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{args.outdir}/groot_resident_fix_timing.json", "w"), indent=1)
    print("DONE_GROOT_RESIDENT_FIX")


if __name__ == "__main__":
    main()
