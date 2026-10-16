#!/usr/bin/env python3
"""(board) Closed-loop policy server for the IQ-9075: one decision per request, exactly the full-cost path of
drone_bframe_board.py (BoardPolicy.decide: GR00T's transforms on the A78 -> raw write -> BFRAME on one NSP -> numpy head).

Speaks the protocol of drone/closed_loop.py --policy HOST:PORT (and drone/drone_fulltrt.py serve): messages are a 4-byte
big-endian length + payload; request payload = JSON header line + b"\\n" + 224*224*3 uint8 RGB bytes.
  {"op": "act", "instr": str, "state": [x, y, z, yaw]} -> {"actions": 16x4, "feat_b64": float32[2048], "nvalid": n,
                                                          "ms": {..., "board_wall"}}
  {"op": "ping"} -> {"ok": 1, "loaded": [...], "n_dec": n}        {"op": "quit"} -> {} and a clean shutdown
board_wall = request fully received -> reply ready (everything on the board; the network is excluded), which
closed_loop.py --d own uses as the decision's latency.

Listens on 127.0.0.1:--port by default; reach it from the simulator host with your own ssh tunnel (ssh -L PORT:127.0.0.1:PORT
<board>), or pass --host 0.0.0.0 on a trusted network. Start it detached (setsid/nohup) so an ssh drop never signals the
process that owns the NPU mappings. Shutdown: op quit -> orchestrator QUIT (SIGTERM only if it hangs). Never SIGKILL.
  $DRONE_IQ9_PY drone/iq9/policy/drone_server_iq9.py --out $DRONE_IQ9_BOARD_DIR/runs/server [--device 0] [--port 47124]
"""
import os, sys, json, time, socket, struct, base64, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import drone_bframe_board as D   # sets the BLAS thread env before numpy loads
import numpy as np


def recv_n(c, n):
    b = bytearray()
    while len(b) < n:
        k = c.recv(n - len(b))
        if not k: raise ConnectionError("peer closed")
        b += k
    return bytes(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--arm", default="fix", choices=["fix", "aaf0"])
    ap.add_argument("--device", type=int, default=0); ap.add_argument("--blas-threads", type=int, default=6)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=47124)
    ap.add_argument("--warmup", type=int, default=3, help="decisions on gate frames before listening")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pol = D.BoardPolicy(a.arm, a.out, device=a.device, blas_threads=a.blas_threads, shm_tag="drone_srv")
    z = np.load(f"{D.ASSETS}/gate_set.npz")
    for k in range(a.warmup): pol.decide(z["images"][k], str(z["instructions"][k]), z["states"][k])
    log = open(f"{a.out}/server.log", "a")
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.host, a.port)); s.listen(1)
    print(f"READY {a.host}:{a.port} arm={a.arm} device={a.device} loaded={pol.R.loaded}", file=log, flush=True)
    print("READY", a.host, a.port, flush=True)
    n_dec = 0; quit_ = False
    try:
        while not quit_:
            c, _ = s.accept()
            try:
                while True:
                    ln = struct.unpack(">I", recv_n(c, 4))[0]; pl = recv_n(c, ln)
                    t_rx = time.perf_counter()
                    hdr, img = pl.split(b"\n", 1); hdr = json.loads(hdr)
                    if hdr["op"] == "quit": quit_ = True; c.sendall(struct.pack(">I", 2) + b"{}"); break
                    if hdr["op"] == "ping":
                        r = json.dumps({"ok": 1, "board": "IQ-9075", "arm": a.arm, "device": a.device, "loaded": pol.R.loaded,
                                        "n_dec": n_dec}).encode()
                        c.sendall(struct.pack(">I", len(r)) + r); continue
                    im = np.frombuffer(img, np.uint8).reshape(224, 224, 3)
                    act, f, nv, ms = pol.decide(im, str(hdr["instr"]), np.asarray(hdr["state"], np.float32))
                    n_dec += 1
                    fb = base64.b64encode(f.astype(np.float32).tobytes()).decode()
                    ms["board_wall"] = (time.perf_counter() - t_rx) * 1e3   # json.dumps below is not included
                    r = json.dumps({"actions": act.tolist(), "feat_b64": fb, "nvalid": int(nv), "ms": ms}).encode()
                    c.sendall(struct.pack(">I", len(r)) + r)
            except ConnectionError as e:
                print(f"connection closed: {e} (n_dec={n_dec})", file=log, flush=True)
            finally:
                c.close()
    finally:
        rc = pol.close(); s.close()
        print(f"DONE n_dec={n_dec} orch_rc={rc}", file=log, flush=True); print("DONE_SERVE", n_dec, flush=True)


if __name__ == "__main__":
    main()
