#!/usr/bin/env python3
"""RESIDENT GR00T N1.5-3B control loop on iq9 (board side; per-graph EXEC + numpy host glue).

Drives the graph-agnostic resident_orch (QnnContext_createFromBinary x7, shared
backend+device, DDR-resident) through the exact GR00T sample_actions pipeline
(action_head_utils.action_head_pytorch_forward + trt_model_forward.eagle_tensorrt_forward).
On-NPU graphs: vit, llm, vlln_vl_self_attention, state_encoder, action_encoder, DiT,
action_decoder. Host glue: host_glue_groot.GrootGlue (mlp1, embed+scatter, posembed,
concat, Euler). 4 denoise steps (num_inference_timesteps=4, dt=+0.25).

REUSES resident_orch.cpp unmodified (its EXEC path is graph-name agnostic).

BOARD-RUN PREREQUISITES:
  - 7 ctx bins deployed to $CTX
  - glue_constants/*.npy deployed to $GLUE
  - per-frame board inputs prepared to $IO as .raw: pixel_values, position_ids,
    input_ids (int), attention_mask, state, embodiment_id, init_actions(noise)
    (these come from the model's OWN transforms on groot_step0.npz, captured on host)
  - golden action (act_5090.npy) on board for the gate
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, sys, time, subprocess, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host_glue_groot import GrootGlue

BROOT = os.environ.get("GROOT_BROOT", _B+"/optionB")
CTX = f"{BROOT}/ctx"; IO = f"{BROOT}/io"; GLUE = f"{BROOT}/glue_constants"
QNN = os.environ.get("QNN_ROOT", _BQ+"")

CTX_BINS = {   # ctx binary path -> its BAKED graph name (resident_orch registers by baked name)
    f"{CTX}/groot_vit_fp16.bin": "groot_vit",
    f"{CTX}/groot_llm_fp16.bin": "groot_llm",
    f"{CTX}/vlln_vl_self_attention_fp16.bin": "vlln_vl_self_attention_47",
    f"{CTX}/state_encoder_fp16.bin": "state_encoder_47",
    f"{CTX}/action_encoder_fp16.bin": "action_encoder_47",
    f"{CTX}/groot_dit_fp16.bin": "groot_dit",
    f"{CTX}/action_decoder_fp16.bin": "action_decoder_47",
}
# logical -> baked graph name (EXEC addresses graphs by their baked name)
GN = {"vit":"groot_vit", "llm":"groot_llm", "vlln":"vlln_vl_self_attention_47",
      "state_encoder":"state_encoder_47", "action_encoder":"action_encoder_47",
      "DiT":"groot_dit", "action_decoder":"action_decoder_47"}
PROTO = ("LOADED", "READY", "OK", "PONG", "BYE", "ERR")
IMAGE_TOKEN_INDEX = int(os.environ.get("GROOT_IMAGE_TOKEN", "151669"))  # CONFIRMED from model.backbone.eagle_model.image_token_index (was guessed 151654)


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
        """inputs: dict tensor_name->raw_path. Returns {out_name: flat float32}, exec_us."""
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


def run_frame(R, g, timers):
    def rec(k, e): timers.setdefault(k, []).append(e)
    # ---- 1. ViT (1 camera: ego_view), 256 patches ----
    o, e = R.exec_graph(GN["vit"], {"pixel_values": f"{IO}/pixel_values.raw",
                                "position_ids": f"{IO}/position_ids.raw"}, "vit"); rec("vit", e)
    vit_embeds = list(o.values())[0].reshape(1, 256, 1152)
    # ---- host: mlp1 projection (+ pixel_shuffle if enabled — verify; absent for this ckpt) ----
    proj = g.mlp1_project(vit_embeds)                                # (1,256,2048)
    # ---- host: embed lookup + image-token scatter ----
    input_ids = np.fromfile(f"{IO}/input_ids.raw", np.int64).reshape(1, -1)
    inputs_embeds = g.embed_and_scatter(input_ids, proj, IMAGE_TOKEN_INDEX)  # (1,N,2048)
    N = inputs_embeds.shape[1]
    w(f"{IO}/_inputs_embeds.raw", inputs_embeds)
    # ---- 2. LLM (Qwen3 12L + eagle_linear) ----
    o, e = R.exec_graph(GN["llm"], {"inputs_embeds": f"{IO}/_inputs_embeds.raw",
                                "attention_mask": f"{IO}/attention_mask.raw"}, "llm"); rec("llm", e)
    backbone = list(o.values())[0].reshape(1, N, 2048)
    w(f"{IO}/_backbone.raw", backbone)
    # ---- 3. vlln + vl_self_attention -> vl_embs ----
    o, e = R.exec_graph(GN["vlln"], {"backbone_features": f"{IO}/_backbone.raw"}, "vlln"); rec("vlln", e)
    vl_embs = list(o.values())[0].reshape(1, -1, 2048)
    w(f"{IO}/_vl_embs.raw", vl_embs)
    # ---- 4. state_encoder (once) ----
    o, e = R.exec_graph(GN["state_encoder"], {"state": f"{IO}/state.raw",
                                          "embodiment_id": f"{IO}/embodiment_id.raw"}, "state"); rec("state", e)
    state_features = list(o.values())[0].reshape(1, 1, 1536)
    # ---- 5. denoise loop: 4 steps, dt=+0.25, Euler ----
    actions = np.fromfile(f"{IO}/init_actions.raw", np.float32).reshape(1, 16, 32)  # SAME noise as golden!
    dit_e = []
    for s in range(g.num_steps):
        td = g.t_discretized(s)                                     # 0,250,500,750
        w(f"{IO}/_actions.raw", actions)
        # baked graph input timesteps_tensor is INT_32; resident_orch reads .raw as FLOAT then
        # castFromFloat<int32_t> -> so write the integer VALUE as float32 (0/250/500/750).
        w(f"{IO}/_ts.raw", np.array([td], np.float32), np.float32)
        o, e = R.exec_graph(GN["action_encoder"], {"actions": f"{IO}/_actions.raw",
                             "timesteps_tensor": f"{IO}/_ts.raw",
                             "embodiment_id": f"{IO}/embodiment_id.raw"}, "aenc"); rec("aenc", e)
        af = g.add_action_pos_embed(list(o.values())[0].reshape(1, 16, 1536))
        sa = g.assemble_sa_embs(state_features, af)                 # (1,49,1536)
        w(f"{IO}/_sa.raw", sa)
        o, e = R.exec_graph(GN["DiT"], {"sa_embs": f"{IO}/_sa.raw", "vl_embs": f"{IO}/_vl_embs.raw",
                                    "timesteps_tensor": f"{IO}/_ts.raw"}, "dit"); dit_e.append(e)
        mo = list(o.values())[0].reshape(1, 49, 1024)
        w(f"{IO}/_mo.raw", mo)
        o, e = R.exec_graph(GN["action_decoder"], {"model_output": f"{IO}/_mo.raw",
                             "embodiment_id": f"{IO}/embodiment_id.raw"}, "adec"); rec("adec", e)
        pred = list(o.values())[0].reshape(1, 49, 32)
        actions = g.euler(actions, pred)
    timers.setdefault("dit_step", []).extend(dit_e)
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--outdir", default=f"{BROOT}/out/resident")
    ap.add_argument("--golden", default=f"{IO}/act_5090.npy")
    args = ap.parse_args()
    g = GrootGlue(GLUE)
    R = Resident(args.outdir)
    print(f"[groot] loaded {len(R.loaded)} graphs: {R.loaded}; ctx load {R.load_us/1e3:.1f} ms", flush=True)
    if len(R.loaded) != 7: print(f"WARNING expected 7 graphs, got {len(R.loaded)}", flush=True)
    walls, timers, last = [], {}, None
    try:
        for fr in range(args.frames):
            t0 = time.perf_counter(); last = run_frame(R, g, timers); wdt = time.perf_counter() - t0
            walls.append(wdt)
            print(f"[frame {fr}] wall={wdt*1000:.1f} ms mean|a|={np.abs(last).mean():.5f}", flush=True)
    finally:
        if last is not None:
            np.save(f"{args.outdir}/actions_groot_resident.npy", last)
        R.close()
    fw = np.array(walls); steady = fw[args.warmup:]
    med = lambda k: (float(np.median(timers[k]))/1e3 if timers.get(k) else None)
    out = {"date": time.strftime("%Y-%m-%d"), "tag": "MEASURED", "frames": args.frames,
           "steady_median_ms": round(float(np.median(steady))*1000, 2),
           "control_hz": round(1.0/float(np.median(steady)), 3),
           "per_graph_ms": {k: med(k) for k in ["vit","llm","vlln","state","aenc","dit_step","adec"]},
           "one_time_ctx_load_ms": round(R.load_us/1e3, 1)}
    if os.path.exists(args.golden):
        gold = np.load(args.golden).astype(np.float64).reshape(1, 16, 32)
        a = last.astype(np.float64)
        rel = float(np.mean(np.abs(a-gold))/(np.mean(np.abs(gold))+1e-12))
        cos = float(np.dot(a.ravel(), gold.ravel())/(np.linalg.norm(a)*np.linalg.norm(gold)+1e-12))
        # 26 real dims (7+7+6+6); last 6 are pad
        rreal = float(np.mean(np.abs(a[...,:26]-gold[...,:26]))/(np.mean(np.abs(gold[...,:26]))+1e-12))
        out["action_gate_vs_golden"] = {"rel_mae_32d": round(rel,5), "rel_mae_26real": round(rreal,5),
            "cos": round(cos,6), "fp16_floor_5e-3": rel < 5e-3, "bf16_gate_0.04": rel < 0.04}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{args.outdir}/groot_resident_timing.json", "w"), indent=1)
    print("DONE_GROOT_RESIDENT_LOOP")


if __name__ == "__main__":
    main()
