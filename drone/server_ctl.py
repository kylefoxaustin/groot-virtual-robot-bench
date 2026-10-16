#!/usr/bin/env python3
"""Ping or stop a drone policy server (drone_fulltrt.py serve).
  python3 drone/server_ctl.py ping HOST:PORT      -> {"ok": 1, "n_dec": ..., "gpu": ..., "trt": ...}
  python3 drone/server_ctl.py quit HOST:PORT      -> the server finishes cleanly (never SIGKILL a process holding the GPU)
Needs only the Python standard library."""
import json, socket, struct, sys

if len(sys.argv) != 3 or sys.argv[1] not in ("ping", "quit"): sys.exit(__doc__)
h, p = sys.argv[2].rsplit(":", 1)
c = socket.create_connection((h, int(p)), timeout=60)
pl = json.dumps({"op": sys.argv[1]}).encode() + b"\n"
c.sendall(struct.pack(">I", len(pl)) + pl)
n = struct.unpack(">I", c.recv(4))[0]; b = b""
while len(b) < n: b += c.recv(n - len(b))
print(b.decode() if sys.argv[1] == "ping" else "quit sent; the server exits after this connection")
