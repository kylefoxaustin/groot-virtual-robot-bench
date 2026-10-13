#!/usr/bin/env python3
"""closed_loop_m3 iq9-in-the-loop server with INJECTED diffusion noise.

GR00T's transforms -> Hexagon v73 fp16 GFRAME loop -> unapply, where the flow-matching x0 (board init_actions raw) is taken from a pre-generated
FIXED noise bank keyed by (episode, call) instead of a torch.randn stream -- so it
is BIT-MATCHED (same fp32 values) to the 5090 reference arm. The client calls the
reset_episode endpoint at each env.reset so `call` restarts per episode.

--zero_image: zero the pixel_values fed to the board (in-loop observation-dependence
null). Everything else (noise, state, instruction) held identical.

Board safety: clean QUIT only (never SIGKILL). Reuses iq9_prep + iq9_board from
../closed_loop_m2.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, _W+"/closed_loop_m2")
from iq9_prep import GrootPrep
from iq9_board import IQ9Board


class IQ9Shim:
    def __init__(self, model_path, work, noise_bank, zero_image=False,
                 data_config="fourier_gr1_arms_waist"):
        self.prep = GrootPrep(model_path, data_config)
        self.board = IQ9Board(os.path.join(work, "board"))
        self.noise = np.load(noise_bank)  # (E, C, 1, 16, 32)
        self.zero_image = zero_image
        self.ep = 0
        self.call = 0
        self._staged_ids = None
        self.work = work
        self.log = []
        assert self.noise.shape[2:] == (1, 16, 32), self.noise.shape
        print(f"[M3-IQ9] board loaded {len(self.board.loaded)} graphs "
              f"ctx_load={self.board.load_us/1e3:.1f}ms; noise {self.noise.shape}; "
              f"zero_image={zero_image}", flush=True)
        print("[M3-IQ9] SERVER READY", flush=True)

    def get_modality_config(self):
        return self.prep.dc.modality_config()

    def reset_episode(self, data):
        self.ep = int(data["episode"])
        self.call = 0
        return {"ok": True, "episode": self.ep}

    def get_action(self, observations):
        t0 = time.time()
        raws, tinfo = self.prep.obs_to_raws(observations)
        ids = raws["input_ids"]
        if self._staged_ids is None or not np.array_equal(ids, self._staged_ids):
            self.board.stage_constants(ids, raws["attention_mask"], raws["position_ids"])
            self._staged_ids = ids.copy()
        pixel = raws["pixel_values"]
        if self.zero_image:
            pixel = np.zeros_like(pixel)
        # INJECTED fixed noise as the flow-matching x0 (bit-matched to 5090 arm)
        noise = np.ascontiguousarray(self.noise[self.ep, self.call], dtype=np.float32)
        action_norm, tm = self.board.gframe(pixel, raws["state"], noise)
        unnorm = self.prep.unapply(action_norm)
        reply = {k: np.ascontiguousarray(np.asarray(v), dtype=np.float32) for k, v in unnorm.items()}
        dt = (time.time() - t0) * 1000.0
        self.log.append({"ep": self.ep, "call": self.call, "server_ms": round(dt, 2),
                         "board_wall_ms": round(tm["_wall_s"] * 1000, 1),
                         "board_gframe_ms": round(tm["_gframe_s"] * 1000, 1),
                         "board_push_ms": round(tm["_push_s"] * 1000, 1),
                         "npu_server_us": tm.get("wall_us"),
                         "action_mean_abs": round(float(np.abs(action_norm).mean()), 5),
                         "tinfo": tinfo})
        self.call += 1
        if len(self.log) % 20 == 0:
            print(f"[M3-IQ9] call {len(self.log)} board_wall={tm['_wall_s']*1000:.0f}ms", flush=True)
        return reply

    def dump_log(self):
        p = os.path.join(self.work, "server_calllog.json")
        json.dump(self.log, open(p, "w"), indent=1)
        return p

    def close(self):
        self.board.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=_gvrb_os.environ.get("GROOT_MODEL", _W+"/models/GR00T-N1.5-3B"))
    ap.add_argument("--data_config", default="fourier_gr1_arms_waist")
    ap.add_argument("--noise_bank", required=True)
    ap.add_argument("--zero_image", action="store_true")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--work", default="/tmp/m3_work_iq9")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    from gr00t.eval.robot import RobotInferenceServer
    shim = IQ9Shim(args.model_path, args.work, args.noise_bank,
                   zero_image=args.zero_image, data_config=args.data_config)
    server = RobotInferenceServer(shim, port=args.port)
    server.register_endpoint("reset_episode", shim.reset_episode)
    try:
        server.run()
    finally:
        p = shim.dump_log()
        print(f"[M3-IQ9] call log -> {p}", flush=True)
        print("[M3-IQ9] health:", shim.board.health(), flush=True)
        shim.close()


if __name__ == "__main__":
    main()
