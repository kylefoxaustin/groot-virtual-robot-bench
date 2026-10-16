#!/usr/bin/env python3
"""(board) Run one stand-in context binary RESIDENT on one NSP through the drone orchestrator's EXEC commands: the context is
loaded once; each call reads the input .raw, executes, writes the float output .raw to /dev/shm. That host-side I/O on the
A78 is part of the cost, as a real scan / fusion service would pay it.
  check : run each input of --list once (+ input 0 again: must be bit-identical); --keep DIR saves outputs.npy + repeat0.npy
          for score_check.py
  burst : --spec n:190 = 190 back-to-back calls (one 50 MP frame of 512 px tiles), cycling the inputs; gap_ms:G sleeps G
          after each call
  hz    : --spec hz:R,t:T = paced at R calls/s for T s;  --spec flat_t:T = back-to-back for T s;  --stop FILE ends either early
--bin / --cmd pick the orchestrator binary and EXEC variant: EXEC (any build) | EXECT (stage timers) | EXECF (output written
with one fwrite) | EXECNOP (the host work of EXEC with graphExecute replaced by usleep(--sleep-us)) | EXECONLY (graphExecute
on the already-populated buffers) -- the last four need resident_orch_drone_g (contention/resident_orch_execf.patch).
Writes JSON with the worker's own [start, end] (after load + warm-up), per-call client wall, server exec_us, per-call server
stage timers when the binary prints them, and /proc/stat per-core busy share over [start, end].
Never SIGKILLs: the orchestrator is stopped with QUIT, then SIGTERM only if it hangs.
usage: npu_worker.py MODE --ctx BIN --graph NAME --device D --list INPUT_LIST --out OUT.json [--spec S] [--bin B] [--cmd C]
       [--sleep-us N] [--started FILE] [--go FILE] [--stop FILE] [--keep DIR] [--warmup N] [--tag T]
env: $DRONE_IQ9_BOARD_DIR/env.sh (IQ9_QNN_ROOT, DRONE_IQ9_BLAS, DRONE_IQ9_GLUE)
"""
import os, sys, time, json, argparse, subprocess, shutil
import numpy as np
for _k in ("DRONE_IQ9_BOARD_DIR", "IQ9_QNN_ROOT", "DRONE_IQ9_BLAS", "DRONE_IQ9_GLUE"):
    if not os.environ.get(_k): sys.exit(f"{_k} is not set -- source $DRONE_IQ9_BOARD_DIR/env.sh")
ROOT = os.environ["DRONE_IQ9_BOARD_DIR"]; QNN = os.environ["IQ9_QNN_ROOT"]; BLAS = os.environ["DRONE_IQ9_BLAS"]; GLUE = os.environ["DRONE_IQ9_GLUE"]
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["check", "burst", "hz"]); ap.add_argument("--ctx", required=True); ap.add_argument("--graph", required=True)
ap.add_argument("--device", type=int, required=True); ap.add_argument("--list", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--spec", default=None); ap.add_argument("--started", default=None); ap.add_argument("--go", default=None)
ap.add_argument("--bin", default=f"{ROOT}/bin/resident_orch_drone"); ap.add_argument("--cmd", default="EXEC"); ap.add_argument("--sleep-us", type=int, default=0)
ap.add_argument("--stop", default=None); ap.add_argument("--keep", default=None); ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--tag", default="w")
a = ap.parse_args()
inputs = [l.strip() for l in open(a.list) if l.strip()]
shm = f"/dev/shm/npuw_{a.tag}_{os.getpid()}"; os.makedirs(shm, exist_ok=True)
lists = []
for i, l in enumerate(inputs):   # one single-line input list per input (EXEC takes a list file)
    p = f"{shm}/in{i}.txt"; open(p, "w").write(l + "\n"); lists.append(p)
env = dict(os.environ); env["LD_LIBRARY_PATH"] = f"{QNN}/lib:" + env.get("LD_LIBRARY_PATH", ""); env["ADSP_LIBRARY_PATH"] = f"{QNN}/skel"
args = [a.bin, "--backend", f"{QNN}/lib/libQnnHtp.so", "--system_library", f"{QNN}/lib/libQnnSystem.so",
        "--log_level", "error", "--blas", BLAS, "--glue", GLUE, "--blas_threads", "1",
        "--device", str(a.device), "--ctx", a.ctx]
err = open(f"{shm}.stderr.log", "w")
p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env, text=True, bufsize=1)
loaded = []
while True:
    l = p.stdout.readline()
    if not l: raise SystemExit("orch died during load: " + open(f"{shm}.stderr.log").read()[-500:])
    l = l.strip()
    if l.startswith("LOADED"): loaded.append(l.split()[1])
    elif l.startswith("READY"): load_us = int(l.split("load_us=")[1].split()[0]); break
    elif l.startswith("ERR"): raise SystemExit(l)
assert a.graph in loaded, (a.graph, loaded)
odir = f"{shm}/o"


STAGES = []


def cmdline(i, cmd):
    if cmd == "EXECONLY": return f"EXECONLY {a.graph}\n"
    if cmd == "EXECNOP": return f"EXECNOP {a.graph} {lists[i % len(lists)]} {odir} {a.sleep_us}\n"
    return f"{cmd} {a.graph} {lists[i % len(lists)]} {odir}\n"


def procstat():
    out = {}
    for l in open("/proc/stat"):
        if l.startswith("cpu"):
            f = l.split(); v = list(map(int, f[1:8])); out[f[0]] = [sum(v), v[3] + v[4]]   # total, idle+iowait jiffies
    return out


def ex(i, cmd=None):
    cmd = cmd or a.cmd
    t0 = time.time(); p.stdin.write(cmdline(i, cmd)); p.stdin.flush()
    while True:
        l = p.stdout.readline()
        if not l: raise RuntimeError("orch closed")
        l = l.strip()
        if l.startswith(("OK ", "ERR")): break
    t1 = time.time()
    if not l.startswith("OK "): raise RuntimeError(l)
    kv = {k: int(v) for k, v in (x.split("=") for x in l.split()[3:] if "=" in x)}
    if kv: STAGES.append(kv)
    return t0, t1, int(l.split()[2])


rec = {"bin": a.bin, "cmd": a.cmd, "sleep_us": a.sleep_us, "mode": a.mode, "graph": a.graph, "device": a.device, "ctx": a.ctx, "loaded": loaded, "ctx_load_ms": load_us / 1e3, "spec": a.spec}
try:
    for k in range(a.warmup): ex(k, "EXEC" if a.cmd in ("EXECONLY", "EXECNOP") and k == 0 else None)   # EXECONLY needs populated inputs
    if a.mode == "check":
        outs = []
        order = list(range(len(lists))) + [0]
        for i in order:
            t0, t1, us = ex(i)
            f = [os.path.join(dp, fn) for dp, _, fs in os.walk(odir) for fn in fs if fn.endswith(".raw")]
            assert len(f) == 1, f
            outs.append(np.fromfile(f[0], np.float32)); os.remove(f[0])
            if a.keep: os.makedirs(a.keep, exist_ok=True); outs[-1].tofile(f"{a.keep}/out_{len(outs) - 1}.raw")
            rec.setdefault("wall_ms", []).append(round((t1 - t0) * 1e3, 3)); rec.setdefault("exec_us", []).append(us)
        if a.keep:
            os.makedirs(a.keep, exist_ok=True)
            np.save(f"{a.keep}/outputs.npy", np.stack(outs[:-1])); np.save(f"{a.keep}/repeat0.npy", outs[-1])
        rec["repeat0_bit_identical"] = bool(np.array_equal(outs[0], outs[-1]))
        rec["output_elems"] = int(outs[0].size)
    else:
        if a.started: open(a.started, "w").write(str(time.time()))
        if a.go:
            while not os.path.exists(a.go): time.sleep(0.005)
        parts = dict(x.split(":") for x in a.spec.split(","))
        gap = float(parts.get("gap_ms", 0)) / 1e3
        stop = lambda: a.stop is not None and os.path.exists(a.stop)
        fire, wall, exus = [], [], []; STAGES.clear(); PS0 = procstat(); T0 = time.time()
        if "n" in parts or "flat" in parts:
            for k in range(int(parts.get("n", parts.get("flat")))):
                t0, t1, us = ex(k); fire.append(t0); wall.append((t1 - t0) * 1e3); exus.append(us)
                if gap: time.sleep(gap)
        elif "flat_t" in parts:
            k = 0
            while time.time() - T0 < float(parts["flat_t"]) and not stop():
                t0, t1, us = ex(k); fire.append(t0); wall.append((t1 - t0) * 1e3); exus.append(us); k += 1
        else:
            hz, dur = float(parts["hz"]), float(parts["t"]); nxt = T0; k = 0
            while time.time() - T0 < dur and not stop():
                t0, t1, us = ex(k); fire.append(t0); wall.append((t1 - t0) * 1e3); exus.append(us); k += 1
                nxt += 1.0 / hz; s = nxt - time.time()
                if s > 0: time.sleep(min(s, max(0.0, T0 + dur - time.time())))
        T1 = time.time(); PS1 = procstat()
        rec["cpu_busy_pct"] = {c: round(100 * (1 - (PS1[c][1] - PS0[c][1]) / max(1, PS1[c][0] - PS0[c][0])), 2) for c in PS0}
        if STAGES:
            rec["stage_us_mean"] = {k: round(float(np.mean([s_[k] for s_ in STAGES])), 1) for k in STAGES[0]}
            rec["stages"] = STAGES[:]
        rec.update({"n": len(fire), "start": T0, "end": T1, "wall_s": round(T1 - T0, 4),
                    "per_inf_ms_mean": round(float(np.mean(wall)), 4), "per_inf_ms_p95": round(float(np.percentile(wall, 95)), 4),
                    "per_inf_ms_max": round(float(np.max(wall)), 4), "exec_ms_mean": round(float(np.mean(exus)) / 1e3, 4),
                    "achieved_hz": round(len(fire) / (T1 - T0), 3), "duty_pct": round(100 * sum(wall) / 1e3 / (T1 - T0), 2), "gap_ms": gap * 1e3,
                    "fire_rel_s": [round(f - T0, 5) for f in fire], "lat_ms": [round(x, 4) for x in wall], "exec_us": exus})
finally:
    try:
        p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=60)
    except Exception:
        p.terminate(); p.wait(timeout=60)   # SIGTERM only; never SIGKILL
    err.close(); rec["orch_rc"] = p.returncode
    shutil.rmtree(shm, ignore_errors=True)
json.dump(rec, open(a.out, "w"), indent=1)
print("WORKER_DONE", a.cmd, a.mode, a.graph, "dev", a.device, "n", rec.get("n", len(rec.get("wall_ms", []))), "per_inf_ms", rec.get("per_inf_ms_mean", rec.get("wall_ms")), "rc", rec["orch_rc"], flush=True)
