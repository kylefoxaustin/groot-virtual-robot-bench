#!/usr/bin/env python3
"""closed_loop_m3: pre-generate two FIXED diffusion-noise banks (A and B).

Each bank is one flow-matching x0 tensor per policy call, shape (1, H, D)=(1,16,32),
for n_episodes x max_calls_per_ep calls. Saved as a single .npy of shape
(n_episodes, max_calls, 1, 16, 32) fp32.

These SAME fp32 values are injected into BOTH backends:
  - 5090 (bf16): via the torch.randn monkeypatch (cast to bf16 at the draw site)
  - iq9  (fp16): via the board GFRAME init_actions raw (fp32 file, cast to fp16 on-board)
so the only remaining difference between backends is the arithmetic precision.

Generated with numpy Generators seeded independently of any torch/cuda RNG so the
banks are reproducible and identical regardless of backend.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--max_calls", type=int, default=50)  # >=45 (720/16); headroom
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--adim", type=int, default=32)
    ap.add_argument("--out_a", default=_W+"/closed_loop_m3/noiseA.npy")
    ap.add_argument("--out_b", default=_W+"/closed_loop_m3/noiseB.npy")
    ap.add_argument("--seed_a", type=int, default=1000)
    ap.add_argument("--seed_b", type=int, default=2000)
    args = ap.parse_args()

    shape = (args.n_episodes, args.max_calls, 1, args.horizon, args.adim)
    A = np.random.default_rng(args.seed_a).standard_normal(shape).astype(np.float32)
    B = np.random.default_rng(args.seed_b).standard_normal(shape).astype(np.float32)
    np.save(args.out_a, A)
    np.save(args.out_b, B)
    # sanity: A != B, standard-normal-ish stats
    print(f"A {A.shape} mean={A.mean():.4f} std={A.std():.4f} -> {args.out_a}")
    print(f"B {B.shape} mean={B.mean():.4f} std={B.std():.4f} -> {args.out_b}")
    diff = float(np.mean(np.abs(A - B)))
    print(f"mean|A-B| = {diff:.4f} (must be >0; banks are distinct)")
    assert diff > 0.1


if __name__ == "__main__":
    main()
