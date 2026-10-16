#!/usr/bin/env python3
"""Collect the drone training set in the simulator: a scripted teacher (proportional pursuit, sim/drone_sim.expert) flies
from a random start to a randomly named pillar; every 5th tick we log the 224 px FPV frame, the drone state [x, y, z, yaw],
the next 16 teacher setpoints [vx, vy, vz, yaw_rate] (the action chunk the head learns to predict) and the instruction.

  python3 drone/train/collect_data.py OUT_DIR [N_EPISODES=120]
  -> OUT_DIR/drone_nav.npz (images, states, action_chunks, instructions, episode) + meta.json
We used 120 episodes (seeds 1000-1119) -> 4,603 frames, with NVIDIA's EGL renderer (MUJOCO_GL=egl). With a different
renderer the frames can differ by 1 LSB, so a head you retrain from your own data will not be bit-identical to ours.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
os.environ.setdefault("MUJOCO_GL", "egl")
import drone_sim as dc


def rollout(sim, color, seed):
    rng = np.random.default_rng(seed); tgt = np.array(dc.TARGETS[color], float)
    pos = np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(1.5, 3.0)])
    yaw = rng.uniform(-np.pi, np.pi)
    P = [pos.copy()]; Y = [yaw]; A = []
    for _ in range(260):                               # fly the whole teacher trajectory first
        a = dc.expert(P[-1], Y[-1], tgt); A.append(a)
        p, y = dc.step_kin(P[-1], Y[-1], a); P.append(p); Y.append(y)
        if np.linalg.norm((tgt - p)[:2]) < dc.REACH_M: break
    n = len(A); frames = []
    for t in range(0, n, 5):                           # every 5th tick
        sim.set(P[t], Y[t]); img = sim.img()
        chunk = np.array([A[min(t + k, n - 1)] for k in range(dc.HORIZON)], np.float32)   # next 16 setpoints (last one repeated)
        frames.append((img, np.array([*P[t], Y[t]], np.float32), chunk))
    return frames, bool(np.linalg.norm((tgt - P[-1])[:2]) < dc.REACH_M)


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    sim = dc.Sim(); colors = list(dc.TARGETS)
    imgs, states, chunks, instrs, epid = [], [], [], [], []; reached_ct = 0
    for e in range(n_ep):
        color = colors[e % 4]
        fr, reached = rollout(sim, color, seed=1000 + e); reached_ct += int(reached)
        for img, st, ch in fr:
            imgs.append(img); states.append(st); chunks.append(ch); instrs.append(f"fly to the {color} pillar"); epid.append(e)
        if e % 20 == 0: print(f"episode {e}/{n_ep} frames={len(imgs)} teacher reached so far={reached_ct}", flush=True)
    chunks = np.stack(chunks)
    np.savez_compressed(f"{out}/drone_nav.npz", images=np.stack(imgs), states=np.stack(states), action_chunks=chunks,
                        instructions=np.array(instrs), episode=np.array(epid))
    meta = {"n_episodes": n_ep, "n_frames": len(imgs), "teacher_reached_pct": round(100 * reached_ct / n_ep, 1), "img": dc.IMG,
            "horizon": dc.HORIZON, "action_keys": ["vx", "vy", "vz", "yaw_rate"], "colors": colors, "renderer": dc.renderer_name(),
            "action_mean": chunks.reshape(-1, 4).mean(0).round(4).tolist(), "action_std": chunks.reshape(-1, 4).std(0).round(4).tolist()}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print("saved", out, json.dumps(meta)); sim.close()


if __name__ == "__main__":
    main()
