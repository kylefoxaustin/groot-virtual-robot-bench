#!/usr/bin/env python3
"""Find a reset strategy that yields bit-identical initial obs for the same seed.
Tries reseeding the robosuite env.rng + global np.random, and ep_meta replay."""
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


def rs_env(wrapped):
    # unwrap MultiStepWrapper -> gym GrootRoboCasaEnv -> RoboCasaEnv(.env) -> robosuite(.env)
    e = wrapped
    for _ in range(8):
        if hasattr(e, "rng") and hasattr(e, "sim"):
            return e
        e = getattr(e, "env", None) or getattr(e, "unwrapped", None)
        if e is None:
            break
    return None


def img(o):
    for k in o:
        if k.startswith("video."):
            return np.asarray(o[k]).astype(np.float64)


def diff(a, b):
    return float(np.max(np.abs(img(a) - img(b))))


def deep_reset_seed(w, seed):
    """Reseed EVERY layer's rng + global np, then reset(seed)."""
    np.random.seed(seed)
    e = w
    for _ in range(8):
        if e is None:
            break
        if hasattr(e, "rng"):
            try:
                e.rng = np.random.default_rng(seed)
            except Exception:
                pass
        e = getattr(e, "env", None)
    return w.reset(seed=seed)


def main():
    w = make()
    # baseline: plain reset(seed) twice
    o1, _ = w.reset(seed=0)
    o2, _ = w.reset(seed=0)
    print("STRAT plain reset(seed=0) x2 -> img maxdiff", diff(o1, o2))

    rse = rs_env(w)
    print("found robosuite env:", type(rse).__name__ if rse else None,
          "has rng", hasattr(rse, "rng"), "has _ep_meta", hasattr(rse, "_ep_meta"))

    # strat A: deep reseed rng + global then reset
    oa1 = deep_reset_seed(w, 0)[0]
    oa2 = deep_reset_seed(w, 0)[0]
    print("STRAT deep-reseed(0) x2 -> img maxdiff", diff(oa1, oa2))

    # strat B: capture ep_meta after a seeded reset, replay it via set_ep_meta
    if rse is not None and hasattr(rse, "get_ep_meta"):
        np.random.seed(0)
        w.reset(seed=0)
        meta = rse.get_ep_meta()
        import copy
        m = copy.deepcopy(meta)
        # reset with a DIFFERENT seed then force meta back
        def reset_with_meta(seed):
            np.random.seed(seed)
            if hasattr(rse, "set_ep_meta"):
                rse.set_ep_meta(copy.deepcopy(m))
            e = w
            for _ in range(8):
                if e is None: break
                if hasattr(e, "rng"): e.rng = np.random.default_rng(seed)
                e = getattr(e, "env", None)
            return w.reset(seed=seed)[0]
        ob1 = reset_with_meta(0)
        ob2 = reset_with_meta(0)
        print("STRAT ep_meta-replay x2 -> img maxdiff", diff(ob1, ob2))
        print("  ep_meta keys:", list(meta.keys())[:20])
    w.close()


if __name__ == "__main__":
    main()
