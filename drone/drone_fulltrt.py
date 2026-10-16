#!/usr/bin/env python3
"""The GR00T drone policy on NVIDIA's full TensorRT path: full-cost timing, the output gate, a back-to-back loop for the
contention runs, and a policy server for closed-loop flights driven from another machine. The decision path is
drone/policy.py (DronePolicy.decide); every mode below uses it unchanged.

  gate     : the 256 gate frames (drone/assets/gate_set.npz) -> pooled features -> head -> executed setpoints; plus a
             zero-image null arm (frames 0, 8, ..., 248), a pad-id swap and a rerun of frame 0 (both must be bit-identical),
             per-stage synced timings (diagnostic) and each engine timed alone. Score it with drone/score_gate.py.
  fullcost : real per-decision cost, raw 224 px frame + prompt + state -> 16 setpoints, host work included.
             --frames 64 --warmup 5 --offset 0|64|128 per run; we ran 3 runs in separate processes (drone/run_fullcost.sh).
  loop     : the fullcost decision path back-to-back for --duration s over the 256 gate frames (contention runs);
             logs each decision's end time (time.time(), shared with the co-tenant workers) and its actions.
  serve    : closed-loop policy server: one decision per request, exactly the fullcost path. Listens on --host:--port
             (default 127.0.0.1:47124; reach it with ssh -L, or pass --host 0.0.0.0 on a trusted network).
             Reply {"actions", "feat_b64", "nvalid", "ms": {..., "board_wall"}}; board_wall = request received ->
             reply ready (the network is excluded). op "quit" stops it cleanly; never SIGKILL a process holding a GPU.

usage: drone_fulltrt.py MODE --out DIR [--eng DIR] [--vlln FILE] [--frames 64 --warmup 5 --offset 0 --tag T]
env:   setup/env.sh + drone/env.sh (GROOT_REPO, GROOT_MODEL, GROOT_STEP_NPZ, DRONE_ENG, DRONE_ASSETS; JETSON=1 on a Jetson)
"""
import os, sys, time, json, argparse, gzip
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # head BLAS single-threaded (spinning BLAS workers steal host cores)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", choices=["gate", "fullcost", "loop", "serve"])
ap.add_argument("--out", required=True)
ap.add_argument("--eng", default=None, help="dir with vit_fp16.engine + llm_fp16.engine (default $DRONE_ENG)")
ap.add_argument("--vlln", default=None, help="289-token vlln engine (default $DRONE_ENG/vlln289/vlln_vl_self_attention.engine)")
ap.add_argument("--assets", default=None, help="dir with gate_set.npz + drone_head.npz (default $DRONE_ASSETS)")
ap.add_argument("--offset", type=int, default=0); ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--tag", default=None)
ap.add_argument("--duration", type=float, default=60.0, help="loop: seconds of back-to-back decisions after warm-up")
ap.add_argument("--ready", default=None, help="loop: file written when warm (the contention orchestrator waits for it)")
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=47124)
a = ap.parse_args(); sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ASSETS = a.assets or os.environ.get("DRONE_ASSETS", os.path.join(HERE, "assets"))
import numpy as np
import policy as P
import drone_metrics as M

os.makedirs(a.out, exist_ok=True)
print("loading policy...", flush=True)
pol = P.DronePolicy("trt", eng=a.eng, vlln=a.vlln, head=os.path.join(ASSETS, "drone_head.npz"))
import torch
z = np.load(os.path.join(ASSETS, "gate_set.npz"))
IMGS, INS, STS = z["images"], [str(x) for x in z["instructions"]], z["states"]
meta = {**pol.meta, "clocks_pre": P.clocks(), "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": a.tag}


def gate():
    n = len(IMGS); S = torch.cuda.synchronize
    pooled = np.zeros((n, 2048), np.float32); tim = []
    null_idx = list(range(0, n, 8)); pooled_null = np.zeros((len(null_idx), 2048), np.float32)
    nrm = [pol.transform(IMGS[j], INS[j]) for j in range(n)]      # GR00T's transforms on the host (not timed here)
    ENGT = {}; import trt_torch
    _ofwd = trt_torch.Engine.forward

    def _tfwd(self, *aa, **k):   # per-engine synced timer (diagnostic only)
        S(); t0 = time.perf_counter(); r = _ofwd(self, *aa, **k); S()
        ENGT[os.path.basename(self.file)] = time.perf_counter() - t0; return r
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(3): pol.feature_from_vl(*pol.vl_inputs(nrm[0]))   # warm
        trt_torch.Engine.forward = _tfwd
        for j in range(n):
            S(); t0 = time.perf_counter()
            vl, nv = pol.vl_inputs(nrm[j]); pooled[j] = pol.feature_from_vl(vl, nv).squeeze(0).cpu().numpy()
            wall = time.perf_counter() - t0; us = lambda k: int(ENGT.get(k, 0) * 1e6); eng = sum(ENGT.values())
            tim.append({"wall_us": int(wall * 1e6), "vit": us("vit_fp16.engine"), "llm": us("llm_fp16.engine"),
                        "vlln": us(os.path.basename(pol.E["vlln289"].file)), "glue_and_io": int((wall - eng) * 1e6)})
            if j % 64 == 0: print(f"[gate] {j} wall {wall * 1e3:.2f} ms", flush=True)
        trt_torch.Engine.forward = _ofwd
        for k, j in enumerate(null_idx):
            vl, nv = pol.vl_inputs(nrm[j], zero_pixels=True); pooled_null[k] = pol.feature_from_vl(vl, nv).squeeze(0).cpu().numpy()
        vl, nv = pol.vl_inputs(nrm[0], padswap=True); ps = pol.feature_from_vl(vl, nv).squeeze(0).cpu().numpy()
        vl, nv = pol.vl_inputs(nrm[0]); r0 = pol.feature_from_vl(vl, nv).squeeze(0).cpu().numpy()

    def standalone(e):   # each engine alone: CUDA events, 200 iterations after 50 warm-up
        ctx = e.execution_context; bufs = []
        for name, shape, dt in e.in_meta + e.out_meta:
            shp = tuple(ctx.get_tensor_shape(name)) if (name, shape, dt) in e.out_meta else tuple(max(1, s) for s in ctx.get_tensor_shape(name))
            x = (torch.zeros(shp, dtype=dt, device="cuda") if dt in (torch.int64, torch.int32) else torch.randn(shp, device="cuda").to(dt))
            ctx.set_tensor_address(name, x.data_ptr()); bufs.append(x)
        s = torch.cuda.current_stream()
        for _ in range(50): ctx.execute_async_v3(s.cuda_stream)
        st, en = torch.cuda.Event(True), torch.cuda.Event(True); st.record()
        for _ in range(200): ctx.execute_async_v3(s.cuda_stream)
        en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / 200
    alone = {t: standalone(e) for t, e in pol.E.items()}
    np.save(f"{a.out}/pooled_real.npy", pooled); np.save(f"{a.out}/pooled_null.npy", pooled_null)
    swap = M.swap_index(INS)
    ex = lambda A: np.round(M.exec_setpoints(A), 6).tolist()
    with gzip.open(f"{a.out}/gate_actions.json.gz", "wt") as f:
        json.dump({"board": a.tag, "units": "executed setpoints: first 4 of 16 steps x [vx, vz, yaw_rate]",
                   "real": ex(pol.head(pooled, STS)), "null_zero_image": ex(pol.head(pooled_null, STS[null_idx])), "null_frames": null_idx,
                   "planted_swap": ex(pol.head(pooled[swap], STS)), "padswap_max_abs": float(np.abs(ps - pooled[0]).max()),
                   "rerun_max_abs": float(np.abs(r0 - pooled[0]).max())}, f)
    out = {**meta, "n": n, "padswap_max_abs": float(np.abs(ps - pooled[0]).max()), "rerun_max_abs": float(np.abs(r0 - pooled[0]).max()),
           "standalone_engine_ms": alone, "clocks_post": P.clocks(), "rows_us": tim}
    json.dump(out, open(f"{a.out}/gate_run.json", "w"), indent=1)
    print("DONE_GATE padswap", out["padswap_max_abs"], "rerun", out["rerun_max_abs"],
          "standalone_ms", json.dumps({k: round(v, 3) for k, v in alone.items()}), flush=True)
    print(f"score it: python3 drone/score_gate.py {a.out}", flush=True)


def fullcost():
    rows, acts = [], []
    for k in range(a.frames + a.warmup):
        j = (a.offset + k) % len(IMGS)
        act, f, n, ms = pol.decide(IMGS[j], INS[j], STS[j])
        if k >= a.warmup:
            rows.append({"j": int(j), **{s: round(v, 4) for s, v in ms.items()}}); acts.append(act)
    tot = np.array([r["total_ms"] for r in rows])
    out = {"board": a.tag, "run": a.tag, "offset": a.offset, "frames": a.frames, "warmup": a.warmup,
           "software": {k: meta.get(k) for k in ("trt", "torch", "gpu", "torch_threads")}, "utc_start": meta["utc_start"],
           "clocks_nvidia_smi_pre_post": [meta["clocks_pre"], P.clocks()], "total_ms_median": float(np.median(tot)), "rows": rows}
    json.dump(out, open(f"{a.out}/fullcost.json", "w"), indent=1)
    with gzip.open(f"{a.out}/fullcost_actions.json.gz", "wt") as f:
        json.dump({"j": [r["j"] for r in rows], "exec": np.round(M.exec_setpoints(np.array(acts)), 6).tolist()}, f)
    print("DONE_FULL", a.tag, "median_ms", round(float(np.median(tot)), 3), "=", round(1e3 / float(np.median(tot)), 1), "Hz",
          json.dumps({s: round(float(np.median([r[s] for r in rows])), 3) for s in ("transform_ms", "gpu_policy_ms", "head_ms")}), flush=True)


def loop():
    def decide(j): return pol.decide(IMGS[j], INS[j], STS[j])[0]
    for k in range(a.warmup): decide(k % len(IMGS))
    if a.ready: open(a.ready, "w").write(str(time.time()))
    ts, tst, js, acts = [], [], [], []; k = 0; T0 = time.time()
    while time.time() - T0 < a.duration:
        j = (a.offset + k) % len(IMGS); s0 = time.time()
        acts.append(decide(j)); te = time.time()
        tst.append(s0); ts.append(te); js.append(j); k += 1
    np.savez(f"{a.out}/loop.npz", ts=np.array(ts), tstart=np.array(tst), j=np.array(js), actions=np.array(acts))
    out = {**meta, "duration_s": a.duration, "n": len(ts), "rate_hz_whole": len(ts) / (ts[-1] - T0),
           "interval_ms_median": float(np.median(np.diff(ts)) * 1e3), "clocks_post": P.clocks()}
    json.dump(out, open(f"{a.out}/loop.json", "w"), indent=1)
    print("DONE_LOOP", a.tag, "n", len(ts), "median_interval_ms", round(out["interval_ms_median"], 3), flush=True)


def serve():
    import socket, struct, base64

    def recv_n(c, n):
        b = bytearray()
        while len(b) < n:
            k = c.recv(n - len(b))
            if not k: raise ConnectionError("peer closed")
            b += k
        return bytes(b)
    for k in range(a.warmup): pol.decide(IMGS[k], INS[k], STS[k])
    log = open(f"{a.out}/server.log", "a")
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.host, a.port)); s.listen(1)
    print(f"READY serve {a.host}:{a.port} {json.dumps(meta)}", file=log, flush=True); print("READY", a.host, a.port, flush=True)
    n_dec = 0; quit_ = False
    while not quit_:
        c, _ = s.accept()
        try:
            while True:
                ln = struct.unpack(">I", recv_n(c, 4))[0]; pl = recv_n(c, ln)
                t_rx = time.perf_counter()
                hdr, img = pl.split(b"\n", 1); hdr = json.loads(hdr)
                if hdr["op"] == "quit": quit_ = True; c.sendall(struct.pack(">I", 2) + b"{}"); break
                if hdr["op"] == "ping":
                    r = json.dumps({"ok": 1, "n_dec": n_dec, "gpu": meta["gpu"], "trt": meta["trt"]}).encode()
                    c.sendall(struct.pack(">I", len(r)) + r); continue
                im = np.frombuffer(img, np.uint8).reshape(224, 224, 3)
                act, f, nv, ms = pol.decide(im, str(hdr["instr"]), np.asarray(hdr["state"], np.float32))
                n_dec += 1
                fb = base64.b64encode(f.astype(np.float32).tobytes()).decode()
                ms["board_wall"] = (time.perf_counter() - t_rx) * 1e3
                r = json.dumps({"actions": act.tolist(), "feat_b64": fb, "nvalid": int(nv), "ms": ms}).encode()
                c.sendall(struct.pack(">I", len(r)) + r)
        except ConnectionError as e:
            print(f"connection closed: {e} (n_dec={n_dec})", file=log, flush=True)
        finally:
            c.close()
    s.close()
    print(f"DONE n_dec={n_dec} clocks_post={P.clocks()}", file=log, flush=True); print("DONE_SERVE", n_dec, flush=True)


{"gate": gate, "fullcost": fullcost, "loop": loop, "serve": serve}[a.mode]()
