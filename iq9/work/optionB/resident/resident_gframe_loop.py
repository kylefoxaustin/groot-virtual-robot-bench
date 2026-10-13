#!/usr/bin/env python3
"""IN-MEMORY resident GR00T N1.5-3B control loop on iq9 (GFRAME command).

Same resident_orch process (all 7 GR00T contexts DDR-resident), but the ENTIRE
glued action loop now runs IN the C++ process via a single `GFRAME` command:
ViT -> [mlp1 GEMM + embed/scatter] -> LLM -> vl-self-attn -> state-enc ->
4x{action-enc + pos-embed/concat -> DiT -> action-dec -> Euler}, with all FLOAT_32
intermediates memcpy'd graph->graph (no per-graph .raw round-trips). Only small
INT_32 file inputs (position_ids/attention_mask) + the file inputs (pixel/state/
time_proj/init_actions) are read once per frame via the verified FLOAT->native cast.

Times GFRAME wall-clock over N frames and gates the action vs the fp32/bf16 golden
AND runs the null arm (zeroed image) AND compares to the file-loop action.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
_BP = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_PY_ROOT", "/opt/iq9-python"))
import os, sys, time, subprocess, json, argparse
import numpy as np

BROOT = os.environ.get("GROOT_BROOT", _B+"/optionB")
CTX = f"{BROOT}/ctx_fix"; IO = f"{BROOT}/io"; GLUE = f"{BROOT}/glue_constants"
QNN = os.environ.get("QNN_ROOT", _BQ+"")
BLAS = os.environ.get("GROOT_BLAS",
    _BP+"/python/lib/python3.12/site-packages/torch/lib/libopenblas.so.0")

CTX_BINS = [
    f"{CTX}/groot_vit_fp16.bin", f"{CTX}/groot_llm_fp16.bin",
    f"{CTX}/vlln_vl_self_attention_fp16.bin", f"{CTX}/state_encoder_fp16.bin",
    f"{CTX}/action_encoder_fp16.bin", f"{CTX}/groot_dit_fp16.bin",
    f"{CTX}/action_decoder_fp16.bin",
]
PROTO = ("LOADED", "READY", "OK", "OKGFRAME", "PONG", "BYE", "ERR")


class Resident:
    def __init__(self, outdir, blas_threads=6, log_level="error"):
        self.outdir = outdir; os.makedirs(outdir, exist_ok=True)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{QNN}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["ADSP_LIBRARY_PATH"] = f"{QNN}/skel"
        here = os.path.dirname(os.path.abspath(__file__))
        args = [f"{here}/resident_orch", "--backend", f"{QNN}/lib/libQnnHtp.so",
                "--system_library", f"{QNN}/lib/libQnnSystem.so", "--log_level", log_level,
                "--blas", BLAS, "--glue", GLUE, "--blas_threads", str(blas_threads)]
        for path in CTX_BINS:
            args += ["--ctx", path]
        self.errlog = open(f"{outdir}/orch_stderr.log", "w")
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errlog, env=env, text=True, bufsize=1)
        self.loaded, self.load_us = [], None
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("resident_orch died during load; see orch_stderr.log")
            line = line.strip()
            if line.startswith("LOADED"): self.loaded.append(line.split()[1])
            elif line.startswith("READY"):
                for t in line.split():
                    if t.startswith("load_us="): self.load_us = int(t.split("=")[1])
                break
            elif line.startswith("ERR"): raise RuntimeError(f"load error: {line}")

    def _readline_proto(self):
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("resident_orch closed stdout")
            line = line.strip()
            if line.startswith(PROTO): return line

    def gframe(self, io_dir, out_file, pixel_path=None):
        cmd = f"GFRAME {io_dir} {out_file}" + (f" {pixel_path}" if pixel_path else "")
        t0 = time.perf_counter()
        self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()
        resp = self._readline_proto()
        wall = time.perf_counter() - t0
        if not resp.startswith("OKGFRAME"):
            raise RuntimeError(f"GFRAME failed: {resp} (see orch_stderr.log)")
        t = {}
        for tokn in resp.split()[1:]:
            k, v = tokn.split("="); t[k] = int(v)
        return wall, t

    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(timeout=30)
        except Exception:
            self.p.terminate()
        self.errlog.close()


def gate_action(a, gold):
    a = a.astype(np.float64).reshape(1, 16, 32); g = gold.astype(np.float64).reshape(1, 16, 32)
    rel = float(np.mean(np.abs(a - g)) / (np.mean(np.abs(g)) + 1e-12))
    cos = float(np.dot(a.ravel(), g.ravel()) / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-12))
    rreal = float(np.mean(np.abs(a[..., :26] - g[..., :26])) / (np.mean(np.abs(g[..., :26])) + 1e-12))
    return {"rel_mae_32d": round(rel, 5), "rel_mae_26real": round(rreal, 5), "cos": round(cos, 6),
            "fp16_floor_5e-3": rel < 5e-3, "bf16_gate_0.04": rel < 0.04}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--outdir", default=f"{BROOT}/out/gframe")
    ap.add_argument("--golden", default=f"{IO}/groot_fresh_golden.npy")
    ap.add_argument("--fileloop-action", default=f"{BROOT}/out/resident_fix/action_final.npy")
    ap.add_argument("--blas-threads", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out_real = f"{args.outdir}/action_gframe.raw"
    out_null = f"{args.outdir}/action_gframe_null.raw"

    # null pixel file
    pv = np.fromfile(f"{IO}/pixel_values.raw", np.float32)
    null_pix = f"{IO}/_pixel_null.raw"; np.zeros_like(pv).tofile(null_pix)

    R = Resident(args.outdir, blas_threads=args.blas_threads)
    print(f"[gframe] loaded {len(R.loaded)} graphs: {R.loaded}; ctx load {R.load_us/1e3:.1f} ms", flush=True)
    if len(R.loaded) != 7: print(f"WARNING expected 7 graphs, got {len(R.loaded)}", flush=True)

    walls, tacc, prev, f2f = [], {}, None, []
    try:
        for fr in range(args.frames):
            wall, t = R.gframe(IO, out_real)  # real image
            walls.append(wall)
            for k, v in t.items(): tacc.setdefault(k, []).append(v)
            cur = np.fromfile(out_real, np.float32).copy()
            if prev is not None: f2f.append(float(np.abs(cur - prev).max()))
            prev = cur
            print(f"[frame {fr}] wall={wall*1000:.1f} ms (server={t['wall_us']/1e3:.1f}) mean|a|={np.abs(cur).mean():.5f}", flush=True)
        # null arm (one frame, zeroed pixel)
        R.gframe(IO, out_null, pixel_path=null_pix)
    finally:
        R.close()

    real = np.fromfile(out_real, np.float32)
    null = np.fromfile(out_null, np.float32)
    np.save(f"{args.outdir}/action_gframe.npy", real.reshape(1, 16, 32))
    gold = np.load(args.golden)

    fw = np.array(walls); steady = fw[args.warmup:]
    med = lambda k: (float(np.median(tacc[k])) / 1e3 if k in tacc else None)
    out = {"date": time.strftime("%Y-%m-%d"), "tag": "MEASURED",
           "title": "GR00T N1.5-3B in-memory GFRAME resident loop on iq9 v73",
           "frames": args.frames, "warmup_excluded": args.warmup, "steady_n": len(steady),
           "frame_wall_ms": {"all": [round(x*1000,1) for x in walls],
                             "frame0_ms": round(walls[0]*1000,1),
                             "steady_median_ms": round(float(np.median(steady))*1000,2),
                             "steady_min_ms": round(float(steady.min())*1000,2),
                             "steady_max_ms": round(float(steady.max())*1000,2),
                             "steady_std_ms": round(float(steady.std())*1000,2)},
           "control_hz_steady_median": round(1.0/float(np.median(steady)), 3),
           "server_wall_ms_median": med("wall_us"),
           "per_graph_execute_ms_median": {"vit": med("vit"), "llm": med("llm"), "vlln": med("vlln"),
               "state": med("state"), "aenc_4step_sum": med("aenc_sum"), "dit_4step_sum": med("dit_sum"),
               "adec_4step_sum": med("adec_sum")},
           "npu_execute_sum_ms_median": round(sum(v for v in [med("vit"),med("llm"),med("vlln"),med("state"),
               med("aenc_sum"),med("dit_sum"),med("adec_sum")] if v), 2),
           "host_glue_ms_median": {"mlp1_gemm": med("mlp1"), "embed_scatter": med("gluevlm"),
               "action_head_glue_4step": med("glueah")},
           "one_time_context_load_ms": round(R.load_us/1e3, 1),
           "correctness_gate_real_vs_golden": gate_action(real, gold),
           "null_arm": {}, "frame_to_frame_action_max_abs_diff": {
               "values": [round(x,8) for x in f2f], "max": (round(max(f2f),8) if f2f else None),
               "note": "identical inputs every frame; nonzero => NPU fp16 run-to-run nondeterminism"}}
    # null arm separation
    a = real.astype(np.float64); n = null.astype(np.float64); g = gold.astype(np.float64).ravel()
    rvn = float(np.mean(np.abs(a-n))/(np.mean(np.abs(g))+1e-12))
    nvg = float(np.mean(np.abs(n-g))/(np.mean(np.abs(g))+1e-12))
    rvg = out["correctness_gate_real_vs_golden"]["rel_mae_32d"]
    out["null_arm"] = {"real_vs_null_rel_mae": round(rvn,5), "null_vs_golden_rel_mae": round(nvg,5),
        "real_vs_golden_rel_mae": rvg, "separation_ratio_over_gate": round(rvn/(rvg+1e-12),1),
        "null_distinguishable": rvn > 10*rvg}
    # vs file-loop action
    if os.path.exists(args.fileloop_action):
        fl = np.load(args.fileloop_action).astype(np.float64).ravel()
        out["vs_fileloop_action"] = {"max_abs_diff": round(float(np.abs(a-fl).max()),8),
            "rel_mae": round(float(np.mean(np.abs(a-fl))/(np.mean(np.abs(fl))+1e-12)),6)}
    else:
        out["vs_fileloop_action"] = "fileloop action not found at " + args.fileloop_action
    print("\n=== GFRAME (in-memory glue) SUMMARY ===")
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{args.outdir}/gframe_timing.json", "w"), indent=1)
    print("DONE_GFRAME_LOOP")


if __name__ == "__main__":
    main()
