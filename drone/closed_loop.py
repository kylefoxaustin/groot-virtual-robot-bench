#!/usr/bin/env python3
"""Closed-loop drone flights: the policy flies the simulated drone to the named pillar, 40 held-out episodes (seeds 7000-7039).

Two ways to run the policy:
  --policy local       GR00T in this process on this machine's GPU (what the RTX 5090 runs used)
  --policy HOST:PORT   a policy server on a board (`drone/drone_fulltrt.py serve`, what the Jetson runs used: the simulator
                       on one machine, every decision made by the board). Use an ssh tunnel or --host on the server.
Schedules (drone/delay_loop.py):
  --K 4                 lockstep: the simulator pauses for every decision, then flies K = 4 setpoints (80 ms). Default.
  --K 15                lockstep, 15 setpoints per decision.
  --d N [--dmode M]     inject a delay of N ticks (20 ms each): the drone keeps flying the previous chunk while deciding.
  --d own               back-to-back at each decision's own measured latency: d = K = ceil(latency / 20 ms). Latency is the
                        board-side wall time (server) or transform-to-actions wall time (local); the network is excluded.
Output: <out>/<arm>_episodes.jsonl (one row per episode) and <arm>_meta.json; compare with ours using
  python3 drone/compare_closed_loop.py <out>/<arm>_episodes.jsonl results/drone/raw/closed_loop/<published arm>.jsonl

  python3 drone/closed_loop.py --out $DRONE_RUNS/cl --arm my_K4 --K 4 --episodes 0-9
Renderer: --renderer llvmpipe (default) makes MuJoCo render with Mesa's software EGL, which is bit-repeatable; our
per-episode outcomes were all rendered that way. --renderer nvidia uses the GPU's EGL (fast, not bit-repeatable).
"""
import os, sys, json, time, math, argparse, socket, struct, base64
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # numpy head single-threaded: spinning BLAS workers steal host cores

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True); ap.add_argument("--arm", default=None)
ap.add_argument("--episodes", default="0-39", help="range, e.g. 0-9")
ap.add_argument("--policy", default="local", help="local, or HOST:PORT of drone_fulltrt.py serve")
ap.add_argument("--backend", choices=["trt", "eager"], default="trt", help="local policy only")
ap.add_argument("--K", type=int, default=4, help="setpoints flown per decision (lockstep)")
ap.add_argument("--d", default="0", help="delay in 20 ms ticks, or 'own' (back-to-back at the measured latency)")
ap.add_argument("--dmode", choices=["naive", "aligned"], default="naive")
ap.add_argument("--renderer", choices=["llvmpipe", "nvidia", "any"], default="llvmpipe")
ap.add_argument("--save-npz", action="store_true", help="also save trajectory, actions and features per episode")
a = ap.parse_args()

os.environ.setdefault("MUJOCO_GL", "egl")
if a.renderer == "llvmpipe":
    v = os.environ.get("DRONE_MESA_EGL_VENDOR", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if not os.path.exists(v): sys.exit(f"{v} not found: install Mesa's EGL (Ubuntu: sudo apt install libegl-mesa0) or set DRONE_MESA_EGL_VENDOR")
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = v
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "sim"))
import numpy as np
import drone_sim as dc
import delay_loop as DL
import drone_head_np as H

sim = dc.Sim()
REND = dc.renderer_name()
if a.renderer != "any":
    want = "llvmpipe" if a.renderer == "llvmpipe" else "NVIDIA"
    assert want in REND, f"renderer is {REND!r}, not {a.renderer}"
head = H.Head(os.path.join(os.environ.get("DRONE_ASSETS", os.path.join(HERE, "assets")), "drone_head.npz"))
os.makedirs(a.out, exist_ok=True)


class Remote:
    """client of drone_fulltrt.py serve: 4-byte big-endian length + JSON header line + 224x224x3 uint8 RGB."""
    def __init__(self, hp):
        h, p = hp.rsplit(":", 1); self.c = socket.create_connection((h, int(p)), timeout=300)

    def _rpc(self, hdr, img=b""):
        pl = json.dumps(hdr).encode() + b"\n" + img
        self.c.sendall(struct.pack(">I", len(pl)) + pl)
        n = struct.unpack(">I", self._recv(4))[0]; return json.loads(self._recv(n))

    def _recv(self, n):
        b = bytearray()
        while len(b) < n:
            k = self.c.recv(n - len(b))
            if not k: raise ConnectionError("server closed")
            b += k
        return bytes(b)

    def ping(self): return self._rpc({"op": "ping"})

    def act(self, img, instr, st):
        r = self._rpc({"op": "act", "instr": instr, "state": [float(x) for x in st]}, img.tobytes())
        return np.array(r["actions"]), np.frombuffer(base64.b64decode(r["feat_b64"]), np.float32), r["ms"]


if a.policy == "local":
    import platform
    if platform.machine() == "aarch64" and not os.environ.get("DRONE_ALLOW_LOCAL_ON_JETSON"):
        sys.exit("On a Jetson, Mesa's EGL and the GPU policy crash when they share one process. Start the policy server "
                 "(drone/drone_fulltrt.py serve) and fly against it with --policy 127.0.0.1:47124 (drone/README.md).")
    import policy as P
    pol = P.DronePolicy(a.backend)
    meta_pol = pol.meta
    _z = np.load(os.path.join(os.environ.get("DRONE_ASSETS", os.path.join(HERE, "assets")), "gate_set.npz"))
    for _k in range(5): pol.decide(_z["images"][_k], str(_z["instructions"][_k]), _z["states"][_k])   # warm-up, as the server does

    def decide(img, instr, st):
        ac, f, n, ms = pol.decide(img, instr, st)
        return ac, f, {"latency_ms": ms["total_ms"]}
else:
    pol = Remote(a.policy); meta_pol = {"server": a.policy, **pol.ping()}

    def decide(img, instr, st):
        t0 = time.perf_counter(); ac, f, ms = pol.act(img, instr, st); rtt = (time.perf_counter() - t0) * 1e3
        return ac, f, {"latency_ms": float(ms["board_wall"]), "network_ms": rtt - float(ms["board_wall"])}


def policy_fn(img, instr, st):
    ac, f, x = decide(img, instr, st)
    x["head_recheck_maxabs"] = float(np.abs(head(f, st)[0] - ac).max())   # host re-check of the head on the returned feature
    x["f"] = f
    return ac, x


if a.d == "own":
    K, d, tag = (lambda dn: dn), (lambda ex: math.ceil(ex["latency_ms"] / DL.DT_TICK_MS)), "own"
else:
    K, d, tag = a.K, int(a.d), a.d
arm = a.arm or (f"backtoback_{a.dmode}" if a.d == "own" else f"K{a.K}_d{a.d}_{a.dmode}")
meta = {"arm": arm, "policy": meta_pol, "K": a.K if a.d != "own" else "= d", "d": tag, "dmode": a.dmode, "renderer": REND,
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(meta, open(f"{a.out}/{arm}_meta.json", "w"), indent=1)
print("META", json.dumps(meta), flush=True)
lo, hi = map(int, a.episodes.split("-"))
jl = open(f"{a.out}/{arm}_episodes.jsonl", "a")
for e in range(lo, hi + 1):
    t0 = time.time()
    row, arr, decs, ex = DL.run_episode(dc, sim, e, policy_fn, K, d, a.dmode)
    ok, nchk, nbad, _ = DL.check_arrivals(arr, arr["dec_d"], a.dmode)
    assert ok, f"chunk arrival check failed in episode {e}"
    lat = np.array([x["latency_ms"] for x in ex])
    row.update({"arm": arm, "renderer": REND, "wall_s": round(time.time() - t0, 1), "policy_ms_median": float(np.median(lat)),
                "latency_ms_max": float(lat.max()), "d_hist": {int(k): int(v) for k, v in zip(*np.unique(arr["dec_d"], return_counts=True))},
                "head_recheck_maxabs": max(x["head_recheck_maxabs"] for x in ex)})
    if "network_ms" in ex[0]: row["network_ms_median"] = float(np.median([x["network_ms"] for x in ex]))
    if a.save_npz:
        arr["feats"] = np.array([x["f"] for x in ex], np.float32); arr["latency_ms"] = lat
        np.savez_compressed(f"{a.out}/{arm}_ep{e:02d}.npz", **arr)
    jl.write(json.dumps(row) + "\n"); jl.flush()
    print(f"[{arm}] ep {e} {row['color']:6s} reached={row['reached']} decisions={row['n_decisions']} final_dist={row['final_dist_m']:.2f} m "
          f"d={row['d_hist']} policy={row['policy_ms_median']:.1f} ms wall={row['wall_s']} s", flush=True)
sim.close()
print("DONE", arm, flush=True)
