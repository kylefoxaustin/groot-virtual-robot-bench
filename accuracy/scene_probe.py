#!/usr/bin/env python3
"""Scene bit-match reference: reproduce deep_reseed + env.reset for each seed and hash
the CALL-0 observation -- NO policy server. Because the call-0 obs is fully determined
by seed+deep_reseed (before any action), these hashes are exactly what the 5090
reference client saw (same client code). Compare against each board run's
call0_obs_sha1 to prove the board saw the same initial scenes."""
import argparse, json
from functools import partial
import gymnasium as gym
import numpy as np
from rollout_client import make_env, deep_reseed, obs_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name",
                    default="gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = gym.vector.SyncVectorEnv([partial(make_env, a.env_name, a.max_episode_steps)])
    hashes = {}
    for ep in range(a.n_episodes):
        seed = a.base_seed + ep
        deep_reseed(env, seed)
        obs, _ = env.reset(seed=seed)
        hashes[str(seed)] = obs_hash(obs)
        print(f"[scene] seed={seed} c0={hashes[str(seed)][:12]}", flush=True)
    env.close()
    json.dump(hashes, open(a.out, "w"), indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
