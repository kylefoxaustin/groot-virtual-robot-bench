#!/usr/bin/env python3
"""M7 calibration-capture server: run the REAL fine-tuned GR00T policy on the 5090 to
drive a RoboCasa rollout (so the observation distribution is the true in-loop one), and
in parallel convert each observation to the SAME board raws the iq9 pipeline consumes
(GrootPrep.obs_to_raws) and save them, subsampled, to a capture dir.

These captured board raws become the REAL calibration set for int8 quantization
(fed through the fp16 ONNX pipeline per-graph). NOT random/synthetic calibration.

Reuses m3_server_5090's noise-injection + prefix strip so the captured obs match M5
in-loop conditions exactly.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, _W+"/closed_loop_m3")
sys.path.insert(0, _W+"/closed_loop_m2")   # iq9_prep
from m3_server_5090 import strip_prefix
from iq9_prep import GrootPrep


class CaptureShim:
    def __init__(self, policy, prep, noise_bank, work, capture_dir, every=3, max_frames=48):
        self.policy = policy
        self.prep = prep
        self.noise = np.load(noise_bank)
        self.size = (1, 16, 32)
        self.ep = 0; self.call = 0; self.gcall = 0
        self.work = work; self.capture_dir = capture_dir
        self.every = every; self.max_frames = max_frames
        self.saved = 0
        os.makedirs(capture_dir, exist_ok=True)
        self._real_randn = torch.randn
        def patched(*a, **kw):
            size = kw.get("size", a[0] if a else None)
            if size is not None and tuple(size) == self.size:
                n = self.noise[self.ep, self.call]
                return torch.from_numpy(np.ascontiguousarray(n)).to(
                    device=kw.get("device", "cpu"), dtype=kw.get("dtype", torch.float32))
            return self._real_randn(*a, **kw)
        torch.randn = patched
        print("[M7-CAP] noise", self.noise.shape, "capture->", capture_dir, flush=True)
        print("[M7-CAP] SERVER READY", flush=True)

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def reset_episode(self, data):
        self.ep = int(data["episode"]); self.call = 0
        return {"ok": True, "episode": self.ep}

    def get_action(self, observations):
        obs = strip_prefix({k: v for k, v in observations.items()})
        if "video.ego_view" not in obs and "video.ego_view_bg_crop_pad_res256_freq20" in obs:
            obs["video.ego_view"] = obs.pop("video.ego_view_bg_crop_pad_res256_freq20")
        # capture board raws (subsampled) BEFORE running the policy
        if self.saved < self.max_frames and (self.gcall % self.every == 0):
            try:
                raws, info = self.prep.obs_to_raws({k: v for k, v in obs.items()})
                fd = os.path.join(self.capture_dir, f"frame_{self.saved:03d}")
                os.makedirs(fd, exist_ok=True)
                np.ascontiguousarray(raws["pixel_values"], np.float32).tofile(f"{fd}/pixel_values.raw")
                np.ascontiguousarray(raws["position_ids"], np.float32).tofile(f"{fd}/position_ids.raw")
                np.ascontiguousarray(raws["input_ids"], np.int64).tofile(f"{fd}/input_ids.raw")
                np.ascontiguousarray(raws["attention_mask"], np.float32).tofile(f"{fd}/attention_mask.raw")
                np.ascontiguousarray(raws["state"], np.float32).tofile(f"{fd}/state.raw")
                json.dump({"ep": self.ep, "call": self.call, "gcall": self.gcall, "info": str(info)},
                          open(f"{fd}/meta.json", "w"))
                self.saved += 1
                print(f"[M7-CAP] saved frame {self.saved}/{self.max_frames} (ep{self.ep} call{self.call})", flush=True)
            except Exception as e:
                print("[M7-CAP] capture err:", e, flush=True)
        self.gcall += 1
        act = self.policy.get_action(obs)
        self.call += 1
        return act

    def dump_log(self):
        return os.path.join(self.work, "noop.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=_gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain"))
    ap.add_argument("--data_config", default="fourier_gr1_arms_waist")
    ap.add_argument("--embodiment_tag", default="gr1")
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--noise_bank", required=True)
    ap.add_argument("--port", type=int, default=5571)
    ap.add_argument("--work", default="/tmp/m7_cap")
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); np.random.seed(0)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    print(f"[M7-CAP] loading policy {args.model_path}", flush=True)
    dc = load_data_config(args.data_config)
    policy = Gr00tPolicy(model_path=args.model_path, modality_config=dc.modality_config(),
                         modality_transform=dc.transform(), embodiment_tag=args.embodiment_tag,
                         denoising_steps=args.denoising_steps)
    prep = GrootPrep(args.model_path, args.data_config)
    shim = CaptureShim(policy, prep, args.noise_bank, args.work, args.capture_dir,
                       every=args.every, max_frames=args.max_frames)
    server = RobotInferenceServer(shim, port=args.port)
    server.register_endpoint("reset_episode", shim.reset_episode)
    server.run()


if __name__ == "__main__":
    main()
