#!/usr/bin/env python3
"""closed_loop_m3 5090 GR00T reference server with INJECTED diffusion noise.

Same load path as GR00T's stock inference server (GR00T-N1.5-3B, cuda+bf16,
denoising_steps=4), BUT:
  * strips the "unlocked_waist: " instruction prefix so the 296-token
    text-preserving instruction is COMMON-MODE with the iq9 board. Same strip as
    closed_loop_m2/iq9_prep.py.
  * injects a pre-generated FIXED noise bank as the flow-matching x0 via a
    torch.randn monkeypatch keyed by (episode, call). One (1,16,32) draw per
    policy call (flow_matching_action_head.py:364). The client calls the
    reset_episode endpoint at each env.reset so `call` restarts per episode.

Deterministic: with injected noise the only randomness left is CUDA kernel
non-determinism; the determinism-control arm measures it (must be ~0).

Run with $GROOT_PY, with $GROOT_REPO (Isaac-GR00T) on PYTHONPATH.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse, json, os, time
import numpy as np
import torch

WAIST_PREFIX = "unlocked_waist: "


def strip_prefix(obs):
    k = "annotation.human.coarse_action"
    if k not in obs:
        return obs
    v = obs[k]
    def s1(x):
        x = x.decode() if isinstance(x, bytes) else str(x)
        return x[len(WAIST_PREFIX):] if x.startswith(WAIST_PREFIX) else x
    if isinstance(v, np.ndarray):
        obs[k] = np.array([s1(x) for x in v.reshape(-1)]).reshape(v.shape)
    elif isinstance(v, (list, tuple)):
        obs[k] = [s1(x) for x in v]
    else:
        obs[k] = s1(v)
    return obs


class RefShim:
    def __init__(self, policy, noise_bank, horizon, adim, work):
        self.policy = policy
        self.noise = np.load(noise_bank)  # (E, C, 1, H, D)
        self.size = (1, horizon, adim)
        self.ep = 0
        self.call = 0
        self.work = work
        self.log = []
        assert self.noise.shape[2:] == (1, horizon, adim), self.noise.shape
        # install the global randn monkeypatch ONCE (reads self.ep/self.call live)
        self._real_randn = torch.randn
        def patched(*a, **kw):
            size = kw.get("size", a[0] if a else None)
            if size is not None and tuple(size) == self.size:
                n = self.noise[self.ep, self.call]  # (1,H,D) fp32
                return torch.from_numpy(np.ascontiguousarray(n)).to(
                    device=kw.get("device", "cpu"),
                    dtype=kw.get("dtype", torch.float32))
            return self._real_randn(*a, **kw)
        torch.randn = patched
        print("[M3-5090] noise bank", self.noise.shape, "injected via randn patch", flush=True)
        print("[M3-5090] SERVER READY", flush=True)

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def reset_episode(self, data):
        self.ep = int(data["episode"])
        self.call = 0
        return {"ok": True, "episode": self.ep}

    def get_action(self, observations):
        t0 = time.time()
        obs = strip_prefix({k: v for k, v in observations.items()})
        if "video.ego_view" not in obs and "video.ego_view_bg_crop_pad_res256_freq20" in obs:
            obs["video.ego_view"] = obs.pop("video.ego_view_bg_crop_pad_res256_freq20")
        act = self.policy.get_action(obs)
        dt = (time.time() - t0) * 1000.0
        self.log.append({"ep": self.ep, "call": self.call, "server_ms": round(dt, 2)})
        self.call += 1
        return act

    def dump_log(self):
        p = os.path.join(self.work, "server_calllog.json")
        json.dump(self.log, open(p, "w"), indent=1)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=_gvrb_os.environ.get("GROOT_MODEL", _W+"/models/GR00T-N1.5-3B"))
    ap.add_argument("--data_config", default="fourier_gr1_arms_waist")
    ap.add_argument("--embodiment_tag", default="gr1")
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--noise_bank", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default="/tmp/m3_work")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    # determinism knobs (best-effort; injected noise removes the sampling randomness)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    print(f"[M3-5090] loading policy {args.model_path} denoise={args.denoising_steps}", flush=True)
    t0 = time.time()
    dc = load_data_config(args.data_config)
    policy = Gr00tPolicy(model_path=args.model_path, modality_config=dc.modality_config(),
                         modality_transform=dc.transform(), embodiment_tag=args.embodiment_tag,
                         denoising_steps=args.denoising_steps)
    ah = policy.model.action_head
    horizon, adim = ah.config.action_horizon, ah.config.action_dim
    dev = next(policy.model.parameters()).device
    dtp = next(policy.model.parameters()).dtype
    print(f"[M3-5090] loaded {time.time()-t0:.1f}s device={dev} dtype={dtp} "
          f"horizon={horizon} adim={adim} denoise={ah.num_inference_timesteps}", flush=True)
    assert (horizon, adim) == (16, 32), (horizon, adim)
    assert ah.num_inference_timesteps == args.denoising_steps

    shim = RefShim(policy, args.noise_bank, horizon, adim, args.work)
    server = RobotInferenceServer(shim, port=args.port)
    server.register_endpoint("reset_episode", shim.reset_episode)
    server.register_endpoint("get_modality_config", shim.get_modality_config, requires_input=False)
    try:
        server.run()
    finally:
        p = shim.dump_log()
        print(f"[M3-5090] call log -> {p}", flush=True)


if __name__ == "__main__":
    main()
