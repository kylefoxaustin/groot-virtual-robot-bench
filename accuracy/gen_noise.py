#!/usr/bin/env python3
"""Pre-generate FIXED diffusion-noise banks for paired closed-loop runs.

The published accuracy runs use ONE bank: seed 5000, shape (20 episodes, 50 calls, 1, 16, 32),
sha256 of the raw fp32 bytes 912f1eef81ba928e916d73d739cc2d85318f8acf31f7c8a4cc24bca2d93067ac:
    python gen_noise.py --n_episodes 20 --seed_a 5000 --out_a noise_bank.npy --only_a


Each bank is one flow-matching x0 tensor per policy call, shape (1, H, D)=(1,16,32),
for n_episodes x max_calls_per_ep calls. Saved as a single .npy of shape
(n_episodes, max_calls, 1, 16, 32) fp32.

These SAME fp32 values are injected into every backend via the torch.randn monkeypatch in
reference_server.py / policy_server.py (cast to the compute dtype at the draw site), so the only
remaining difference between backends is the arithmetic.

Generated with numpy Generators seeded independently of any torch/cuda RNG so the
banks are reproducible and identical regardless of backend.
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--max_calls", type=int, default=50)  # >=45 (720/16); headroom
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--adim", type=int, default=32)
    ap.add_argument("--out_a", default="noiseA.npy")
    ap.add_argument("--out_b", default="noiseB.npy")
    ap.add_argument("--only_a", action="store_true", help="write bank A only (the published single-bank setup)")
    ap.add_argument("--seed_a", type=int, default=1000)
    ap.add_argument("--seed_b", type=int, default=2000)
    args = ap.parse_args()

    shape = (args.n_episodes, args.max_calls, 1, args.horizon, args.adim)
    A = np.random.default_rng(args.seed_a).standard_normal(shape).astype(np.float32)
    B = np.random.default_rng(args.seed_b).standard_normal(shape).astype(np.float32)
    np.save(args.out_a, A)
    import hashlib
    print(f"A sha256(raw fp32) = {hashlib.sha256(A.tobytes()).hexdigest()}")
    if args.only_a:
        print(f"A {A.shape} -> {args.out_a}")
        return
    np.save(args.out_b, B)
    # sanity: A != B, standard-normal-ish stats
    print(f"A {A.shape} mean={A.mean():.4f} std={A.std():.4f} -> {args.out_a}")
    print(f"B {B.shape} mean={B.mean():.4f} std={B.std():.4f} -> {args.out_b}")
    diff = float(np.mean(np.abs(A - B)))
    print(f"mean|A-B| = {diff:.4f} (must be >0; banks are distinct)")
    assert diff > 0.1


if __name__ == "__main__":
    main()
