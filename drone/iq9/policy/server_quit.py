#!/usr/bin/env python3
"""Stop a drone policy server cleanly (drone_server_iq9.py or drone/drone_fulltrt.py serve): sends op "quit", the server
stops its orchestrator with QUIT and exits. Use this instead of killing it.   python3 drone/iq9/policy/server_quit.py HOST:PORT
Needs only the Python standard library."""
import json, socket, struct, sys

if len(sys.argv) != 2 or ":" not in sys.argv[1]: sys.exit(__doc__)
h, p = sys.argv[1].rsplit(":", 1)
c = socket.create_connection((h, int(p)), timeout=120)
pl = json.dumps({"op": "quit"}).encode() + b"\n"
c.sendall(struct.pack(">I", len(pl)) + pl)
n = struct.unpack(">I", c.recv(4))[0]; print("server replied", c.recv(n).decode() or "{}")
