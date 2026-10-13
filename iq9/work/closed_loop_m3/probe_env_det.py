#!/usr/bin/env python3
"""Isolate env/render determinism: reset the SAME seed twice, compare obs (esp. the
rendered image + state). Saves one obs for the policy-determinism probe."""
import argparse, pickle
from functools import partial
import numpy as np
import gymnasium as gym
import robosuite  # noqa
import robocasa  # noqa
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper

ENV = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"


def make():
    e = gym.make(ENV, enable_render=True)
    return MultiStepWrapper(e, video_delta_indices=np.array([0]),
                            state_delta_indices=np.array([0]), n_action_steps=16,
                            max_episode_steps=720)


def img(o):
    for k in o:
        if k.startswith("video."):
            return np.asarray(o[k]).astype(np.float64)
    return None


def state(o):
    return np.concatenate([np.asarray(o[k]).ravel() for k in sorted(o) if k.startswith("state.")])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    # two independent envs, same seed
    e1 = gym.vector.SyncVectorEnv([partial(make)])
    o1, _ = e1.reset(seed=0)
    e2 = gym.vector.SyncVectorEnv([partial(make)])
    o2, _ = e2.reset(seed=0)
    i1, i2 = img(o1), img(o2)
    s1, s2 = state(o1), state(o2)
    print("img shape", i1.shape, "state shape", s1.shape)
    print("IMG maxabsdiff  ", float(np.max(np.abs(i1 - i2))), "meanabsdiff", float(np.mean(np.abs(i1 - i2))))
    print("STATE maxabsdiff", float(np.max(np.abs(s1 - s2))))
    # same env reset twice
    o1b, _ = e1.reset(seed=0)
    print("IMG same-env reset-twice maxabsdiff", float(np.max(np.abs(img(o1b) - i1))))
    pickle.dump({k: (np.asarray(v) if not isinstance(v, (str, list)) else v) for k, v in o1.items()},
                open(a.out, "wb"))
    e1.close(); e2.close()
    print("saved", a.out)


if __name__ == "__main__":
    main()
