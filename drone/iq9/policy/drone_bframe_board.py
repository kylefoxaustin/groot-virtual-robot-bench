#!/usr/bin/env python3
"""(board) The GR00T drone policy on the IQ-9075: output gate and full-cost timing.

Policy: SigLIP ViT (NPU) -> mlp1 GEMM + embedding/image-token scatter (A78, inside the orchestrator) -> Qwen LLM (NPU, the
GR00T port's 296-token build, right-padded) -> vlln + vl_self_attention (NPU, the 289-token build) -> mean-pool (A78), all in
one BFRAME call of the resident orchestrator on one NSP -> the drone head in numpy float64 (A78, single-threaded BLAS).

  gate     : the 256 gate frames written by port/capture_gate_inputs.py (the GPU path's exact eagle inputs) -> pooled
             features -> head -> executed setpoints; a zero-image null arm (frames 0, 8, ..., 248), a pad-id swap and a rerun
             of frame 0 (both must be bit-identical). Writes gate_actions.json.gz in the format drone/score_gate.py scores.
  fullcost : real per-decision cost from the raw 224 px frame + prompt + state of drone/assets/gate_set.npz: GR00T's own
             transforms on the A78 (torch, 1 thread), raw write to /dev/shm, BFRAME, head. --device pins the NSP;
             --go-file holds the first timed decision until the file exists (two streams on two NSPs released together).
             Score with policy/score_fullcost_iq9.py.

  $DRONE_IQ9_PY drone/iq9/policy/drone_bframe_board.py gate     --out DIR [--arm fix|aaf0] [--frames-dir DIR]
  $DRONE_IQ9_PY drone/iq9/policy/drone_bframe_board.py fullcost --out DIR [--arm fix] [--device 0] [--frames 64 --warmup 3 --offset 0]
env: $DRONE_IQ9_BOARD_DIR/env.sh (written by drone/iq9/stage_board.sh). The orchestrator is stopped with QUIT, then SIGTERM
only if it hangs. Never SIGKILL a process that holds the NSP.
"""
import os, sys, time, json, argparse, shutil, gzip, subprocess
# The head's small matmuls run in THIS process. A multi-threaded numpy OpenBLAS leaves its workers spin-waiting after each
# call, and they steal the cores the orchestrator's mlp1 GEMM needs a moment later. Set before numpy loads; removed from
# the orchestrator's environment (its OpenBLAS sizes its own pool, --blas_threads).
_BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
if os.environ.get("DRONE_HEAD_MT") != "1":
    for _k in _BLAS_ENV: os.environ[_k] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRONE = os.path.abspath(os.path.join(HERE, "..", ".."))           # drone/ (drone_head_np.py, drone_metrics.py, assets/)
sys.path.insert(0, DRONE)
import drone_head_np as H
import drone_metrics as M


def _env(k, default=None):
    v = os.environ.get(k, default)
    if v is None: sys.exit(f"{k} is not set -- source $DRONE_IQ9_BOARD_DIR/env.sh (drone/iq9/stage_board.sh writes it)")
    return v


B = _env("DRONE_IQ9_BOARD_DIR")
QNN = _env("IQ9_QNN_ROOT"); BLAS = _env("DRONE_IQ9_BLAS"); GLUE = _env("DRONE_IQ9_GLUE"); CONST = _env("DRONE_IQ9_CONST")
ORCH = os.environ.get("DRONE_IQ9_ORCH", f"{B}/bin/resident_orch_drone")
VIT = {"fix": _env("DRONE_IQ9_VIT_FIX"), "aaf0": _env("DRONE_IQ9_VIT_AAF0")}
LLM = _env("DRONE_IQ9_LLM")
VLLN = {"fix": f"{B}/ctx289/vlln289_fix_fp16.bin", "aaf0": f"{B}/ctx289/vlln289_aaf0_fp16.bin"}
ASSETS = os.environ.get("DRONE_ASSETS", f"{DRONE}/assets")
GROOT = os.environ.get("DRONE_IQ9_GROOT_DIR", f"{B}/groot")      # Isaac-GR00T/, shims/, metadata.json, groot_step0.npz
NPAD, NV, PAD_ID = 296, 289, 151643


class Orch:
    """the resident orchestrator with the drone's three graphs loaded once (ViT, LLM, vlln289)"""
    def __init__(self, arm, outdir, device=None, blas_threads=6, binary=None):
        env = {k: v for k, v in os.environ.items() if k not in _BLAS_ENV}
        env["LD_LIBRARY_PATH"] = f"{QNN}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["ADSP_LIBRARY_PATH"] = f"{QNN}/skel"
        args = [binary or ORCH, "--backend", f"{QNN}/lib/libQnnHtp.so", "--system_library", f"{QNN}/lib/libQnnSystem.so",
                "--log_level", "error", "--blas", BLAS, "--glue", GLUE, "--blas_threads", str(blas_threads)]
        if device is not None: args += ["--device", str(device)]
        for p in (VIT[arm], LLM, VLLN[arm]):
            if not os.path.exists(p): sys.exit(f"missing context binary {p}")
            args += ["--ctx", p]
        os.makedirs(outdir, exist_ok=True)
        self.err = open(f"{outdir}/orch_stderr.log", "w")
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, env=env, text=True, bufsize=1)
        self.loaded = []
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError(f"orchestrator died during load, see {outdir}/orch_stderr.log")
            l = l.strip()
            if l.startswith("LOADED"): self.loaded.append(l.split()[1])
            elif l.startswith("READY"): self.load_us = int(l.split("load_us=")[1].split()[0]); break
            elif l.startswith("ERR"): raise RuntimeError(l)

    def bframe(self, fdir, out, dump=False, pix=None):
        cmd = f"BFRAME {fdir} {CONST} {out} {1 if dump else 0}" + (f" {pix}" if pix else "")
        t0 = time.perf_counter(); self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("orchestrator closed")
            l = l.strip()
            if l.startswith(("OKBFRAME", "ERR")): break
        w = time.perf_counter() - t0
        if not l.startswith("OKBFRAME"): raise RuntimeError(l)
        return w, {k: int(v) for k, v in (x.split("=") for x in l.split()[1:])}

    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(timeout=60)
        except Exception:
            self.p.terminate(); self.p.wait(timeout=60)   # SIGTERM only; never SIGKILL
        self.err.close()
        return self.p.returncode



class BoardPolicy:
    """GR00T's transforms on the A78 + BFRAME + numpy head: one full-cost decision per call (as drone/policy.py decide)."""
    def __init__(self, arm, outdir, device=None, blas_threads=6, shm_tag="drone"):
        import torch
        torch.set_num_threads(1)
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        for p in (f"{GROOT}/shims", f"{GROOT}/Isaac-GR00T"):
            if p not in sys.path: sys.path.insert(0, p)
        import cv2
        from gr00t.experiment.data_config import load_data_config
        from gr00t.data.schema import DatasetMetadata
        from gr00t.model.policy import unsqueeze_dict_values
        self.cv2, self.unsq = cv2, unsqueeze_dict_values
        dc = load_data_config("fourier_gr1_arms_only"); self.tf = dc.transform(); self.tf.eval()
        self.tf.set_metadata(DatasetMetadata.model_validate(json.load(open(f"{GROOT}/metadata.json"))["gr1"]))
        self.base = {k: v for k, v in np.load(f"{GROOT}/groot_step0.npz", allow_pickle=True).items() if not k.startswith("action.")}
        self.head = H.Head(f"{ASSETS}/drone_head.npz")
        self.shm = f"/dev/shm/{shm_tag}_{os.getpid()}"; os.makedirs(self.shm, exist_ok=True)
        self.R = Orch(arm, outdir, device=device, blas_threads=blas_threads)
        self.torch_threads = torch.get_num_threads()

    def decide(self, img224, instr, state):
        """raw 224 px frame + prompt + state -> (actions [16, 4] float64, pooled feature [2048] f32, n_valid, stage ms)"""
        t0 = time.perf_counter()
        o = dict(self.base)
        o["video.ego_view"] = self.cv2.resize(img224, (256, 256)).reshape(1, 256, 256, 3).astype(np.uint8)
        o["annotation.human.action.task_description"] = np.array([str(instr)])
        c = self.unsq(o)
        for k, v in c.items():
            if not isinstance(v, np.ndarray): c[k] = np.array(v)
        n = self.tf(c)
        t1 = time.perf_counter()
        ids = np.asarray(n["eagle_input_ids"]).reshape(-1).astype(np.int64)
        pv = np.asarray(n["eagle_pixel_values"], dtype=np.float32).reshape(-1)
        nv = len(ids); ip = np.full(NPAD, PAD_ID, np.int64); ip[:nv] = ids
        am = np.zeros(NPAD, np.float32); am[:nv] = 1
        ip.tofile(f"{self.shm}/input_ids.raw"); am.tofile(f"{self.shm}/attention_mask.raw"); pv.tofile(f"{self.shm}/pixel_values.raw")
        t2 = time.perf_counter()
        w, srv = self.R.bframe(self.shm, f"{self.shm}/out")
        f = np.fromfile(f"{self.shm}/out.pool", np.float32)
        t3 = time.perf_counter()
        act = self.head(f, np.asarray(state, np.float32))[0]
        t4 = time.perf_counter()
        ms = {"transform_ms": (t1 - t0) * 1e3, "rawwrite_ms": (t2 - t1) * 1e3, "npu_policy_ms": (t3 - t2) * 1e3,
              "head_ms": (t4 - t3) * 1e3, "total_ms": (t4 - t0) * 1e3,
              **{f"srv_{k[:-3] if k.endswith('_us') else k}_ms": v / 1e3 for k, v in srv.items()}}
        return act, f, nv, ms

    def close(self):
        rc = self.R.close(); shutil.rmtree(self.shm, ignore_errors=True); return rc


def gate(a):
    od = a.out; os.makedirs(od, exist_ok=True)
    fr = a.frames_dir or f"{B}/gate_frames"
    frames = sorted(d for d in os.listdir(fr) if d.isdigit())
    z = np.load(f"{ASSETS}/gate_set.npz"); STS = z["states"]; INS = [str(x) for x in z["instructions"]]
    assert len(frames) == len(STS), (len(frames), len(STS))
    head = H.Head(f"{ASSETS}/drone_head.npz")
    R = Orch(a.arm, od, device=a.device)
    pooled = np.zeros((len(frames), 2048), np.float32); tim = []
    null_idx = list(range(0, len(frames), 8)); pooled_null = np.zeros((len(null_idx), 2048), np.float32)
    zero = f"{od}/_zero_pixels.raw"; np.zeros(3 * 224 * 224, np.float32).tofile(zero)
    try:
        for j, f in enumerate(frames):
            w, t = R.bframe(f"{fr}/{f}", f"{od}/f{j:04d}", dump=(j < 4))   # dump: per-token LLM / vlln rows of 4 frames
            pooled[j] = np.fromfile(f"{od}/f{j:04d}.pool", np.float32); t["client_wall_us"] = int(w * 1e6); tim.append(t)
            if j % 64 == 0: print(f"[gate {a.arm}] frame {j}", flush=True)
        for k, j in enumerate(null_idx):
            R.bframe(f"{fr}/{frames[j]}", f"{od}/null{k:03d}", pix=zero)
            pooled_null[k] = np.fromfile(f"{od}/null{k:03d}.pool", np.float32)
        pd = f"{od}/_padswap"; os.makedirs(pd, exist_ok=True)     # same frame 0, pads 151643 -> 0 (mask unchanged)
        for fn in ("pixel_values.raw", "attention_mask.raw"): shutil.copy(f"{fr}/{frames[0]}/{fn}", pd)
        ids = np.fromfile(f"{fr}/{frames[0]}/input_ids.raw", np.int64); m = np.fromfile(f"{pd}/attention_mask.raw", np.float32)
        ids2 = ids.copy(); ids2[m == 0] = 0; ids2.tofile(f"{pd}/input_ids.raw")
        R.bframe(pd, f"{od}/padswap")
        R.bframe(f"{fr}/{frames[0]}", f"{od}/rerun0")                 # determinism
    finally:
        rc = R.close()
    ps = np.fromfile(f"{od}/padswap.pool", np.float32); r0 = np.fromfile(f"{od}/rerun0.pool", np.float32)
    np.save(f"{od}/pooled_real.npy", pooled); np.save(f"{od}/pooled_null.npy", pooled_null)
    swap = M.swap_index(INS)
    ex = lambda A: np.round(M.exec_setpoints(A), 6).tolist()
    with gzip.open(f"{od}/gate_actions.json.gz", "wt") as f:
        json.dump({"board": a.tag or f"IQ-9075_{a.arm}", "units": "executed setpoints: first 4 of 16 steps x [vx, vz, yaw_rate]",
                   "real": ex(head(pooled, STS)), "null_zero_image": ex(head(pooled_null, STS[null_idx])), "null_frames": null_idx,
                   "planted_swap": ex(head(pooled[swap], STS)), "padswap_max_abs": float(np.abs(ps - pooled[0]).max()),
                   "rerun_max_abs": float(np.abs(r0 - pooled[0]).max())}, f)
    out = {"arm": a.arm, "device": a.device, "frames_dir": fr, "loaded": R.loaded, "ctx_load_ms": R.load_us / 1e3, "orch_rc": rc,
           "n": len(frames), "padswap_max_abs": float(np.abs(ps - pooled[0]).max()), "rerun_max_abs": float(np.abs(r0 - pooled[0]).max()),
           "rows_us": tim}
    json.dump(out, open(f"{od}/gate_run.json", "w"), indent=1)
    print("DONE_GATE", a.arm, "padswap", out["padswap_max_abs"], "rerun", out["rerun_max_abs"], flush=True)
    print(f"score it (host, after copying {od} back): python3 drone/score_gate.py <dir>", flush=True)


def fullcost(a):
    z = np.load(f"{ASSETS}/gate_set.npz")
    IMGS, INS, STS = z["images"], [str(x) for x in z["instructions"]], z["states"]
    os.makedirs(a.out, exist_ok=True)
    pol = BoardPolicy(a.arm, a.out, device=a.device, blas_threads=a.blas_threads, shm_tag="drone_full")
    print("READY", a.tag or a.out, flush=True)
    if a.go_file:
        while not os.path.exists(a.go_file): time.sleep(0.005)
    rows, acts, nvs = [], [], []
    try:
        for k in range(a.frames + a.warmup):
            j = (a.offset + k) % len(IMGS)
            t0 = time.time()
            act, f, nv, ms = pol.decide(IMGS[j], INS[j], STS[j])
            if k >= a.warmup:
                rows.append({"j": int(j), "t_start": t0, "t_end": time.time(), **{s: round(v, 4) for s, v in ms.items()}})
                acts.append(act); nvs.append(int(nv))
    finally:
        rc = pol.close()
    tot = np.array([r["total_ms"] for r in rows])
    out = {"board": "IQ-9075", "run": a.tag, "arm": a.arm, "device": a.device, "frames": a.frames, "warmup": a.warmup, "offset": a.offset,
           "software": {"qairt": os.environ.get("QAIRT_VER", "2.47.0.260601"), "torch_threads": pol.torch_threads,
                        "blas_threads": a.blas_threads, "loaded": pol.R.loaded},
           "orch_rc": rc, "nvalid_unique": sorted(set(nvs)), "total_ms_median": float(np.median(tot)), "rows": rows}
    json.dump(out, open(f"{a.out}/fullcost.json", "w"), indent=1)
    with gzip.open(f"{a.out}/fullcost_actions.json.gz", "wt") as f:
        json.dump({"j": [r["j"] for r in rows], "exec": np.round(M.exec_setpoints(np.array(acts)), 6).tolist()}, f)
    print("DONE_FULL", a.tag or a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gate", "fullcost"])
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default=None)
    ap.add_argument("--arm", default="fix", choices=["fix", "aaf0"], help="ViT + vlln build: default fusion or aaf0")
    ap.add_argument("--device", type=int, default=None, help="HTP device id to pin (strict); default unpinned")
    ap.add_argument("--frames-dir", default=None, help="gate: frames from capture_gate_inputs.py (default $DRONE_IQ9_BOARD_DIR/gate_frames)")
    ap.add_argument("--frames", type=int, default=64); ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--blas-threads", type=int, default=6)
    ap.add_argument("--go-file", default=None)
    a = ap.parse_args()
    gate(a) if a.mode == "gate" else fullcost(a)
