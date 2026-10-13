#!/usr/bin/env python3
"""Closed-loop RoboCasa GR1 rollout client (runs in the robocasa venv on the x86 host).

Backend-agnostic: talks ZMQ to a GR00T policy server (reference_server.py on the 5090, or
policy_server.py on a Jetson / the 5090). Replicates Isaac-GR00T gr00t/eval/simulation.py
(MultiStepWrapper, n_action_steps=16, video/state delta [0]) and adds three things for
paired, cross-board comparisons:
  * deep_reseed: robosuite seeds its scene-generation RNG once at construction, so a plain
    reset(seed) gives a DIFFERENT scene each time. Reseeding every wrapper layer's .rng plus
    the global np.random BEFORE reset makes the initial scene bit-identical.
  * reset_episode endpoint at every env.reset, so the server restarts its injected-noise call
    index per episode -> bit-matched (episode, call) noise pairing across backends.
  * a SHA1 of the CALL-0 observation per episode: call-0 is fully determined by
    seed + deep_reseed (before any policy action), so matching hashes prove two backends saw
    the same initial scene (compare with scene_probe.py)."""
import argparse, hashlib, io, json, time
from functools import partial
from pathlib import Path

import gymnasium as gym
import msgpack
import numpy as np
import zmq

import robosuite  # noqa: F401
import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper


class MsgSerializer:
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=MsgSerializer._enc)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=MsgSerializer._dec)

    @staticmethod
    def _dec(obj):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _enc(obj):
        if isinstance(obj, np.ndarray):
            out = io.BytesIO()
            np.save(out, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": out.getvalue()}
        return obj


class ActionClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=180000):
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.sock.send(MsgSerializer.to_bytes(req))
        resp = MsgSerializer.from_bytes(self.sock.recv())
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self):
        return self._call("ping", requires_input=False)

    def reset_episode(self, ep):
        return self._call("reset_episode", {"episode": int(ep)})

    def get_action(self, observations):
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20")
        return self._call("get_action", observations)


def make_env(env_name, max_episode_steps):
    env = gym.make(env_name, enable_render=True)
    env = MultiStepWrapper(env, video_delta_indices=np.array([0]),
                           state_delta_indices=np.array([0]), n_action_steps=16,
                           max_episode_steps=max_episode_steps)
    return env


def deep_reseed(vec_env, seed):
    np.random.seed(seed)
    for sub in vec_env.envs:
        e = sub
        for _ in range(8):
            if e is None:
                break
            if hasattr(e, "rng"):
                try:
                    e.rng = np.random.default_rng(seed)
                except Exception:
                    pass
            e = getattr(e, "env", None)


def obs_hash(obs):
    """Deterministic hash of the observation (sorted keys, raw ndarray bytes)."""
    h = hashlib.sha1()
    for k in sorted(obs.keys()):
        v = obs[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def flatten_action(action_dict):
    order = ["action.left_arm", "action.right_arm", "action.left_hand",
             "action.right_hand", "action.waist"]
    keys = [k for k in order if k in action_dict] + \
           [k for k in action_dict if k not in order]
    arrs, dims = [], {}
    for k in keys:
        a = np.asarray(action_dict[k])
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        arrs.append(a)
        dims[k] = int(a.shape[-1])
    return np.concatenate(arrs, axis=-1), keys, dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name",
                    default="gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = ActionClient(host=args.host, port=args.port)
    print("[client] ping:", client.ping(), flush=True)
    env = gym.vector.SyncVectorEnv([partial(make_env, args.env_name, args.max_episode_steps)])

    prog = open(out / "progress.jsonl", "a")   # one line per FINISHED episode (inspectable)

    def write_summary():
        n_s = sum(1 for e in episodes if e["success"])
        summ = {"env_name": args.env_name, "n_episodes": args.n_episodes,
                "n_completed": len(episodes), "n_success": n_s,
                "success_rate": n_s / max(1, len(episodes)),
                "max_episode_steps": args.max_episode_steps, "base_seed": args.base_seed,
                "seeds": [args.base_seed + i for i in range(args.n_episodes)],
                "episodes": episodes,
                "action_net_ms_overall_median": float(np.median(all_ms)) if all_ms else None,
                "n_action_calls": len(all_ms)}
        with open(out / "rollout_summary.json", "w") as f:
            json.dump(summ, f, indent=2)
        return summ

    episodes, all_ms = [], []
    for ep in range(args.n_episodes):
        seed = args.base_seed + ep
        try:
            deep_reseed(env, seed)          # bit-identical initial scene across arms
            obs, info = env.reset(seed=seed)
            c0hash = obs_hash(obs)          # scene bit-match probe: CALL-0 obs (policy-independent)
            print("[client] reset_episode:", client.reset_episode(ep), flush=True)
            ep_actions, ep_reward, ep_succ, ep_ms = [], [], [], []
            n_big, ep_success = 0, False
            keys_ref, dims_ref = None, None
            while True:
                t0 = time.time()
                action_dict = client.get_action(obs)
                ep_ms.append((time.time() - t0) * 1000.0)
                actions = action_dict["actions"] if "actions" in action_dict else action_dict
                flat, keys, dims = flatten_action(actions)
                if keys_ref is None:
                    keys_ref, dims_ref = keys, dims
                next_obs, rewards, terms, truncs, env_infos = env.step(actions)
                r = float(np.asarray(rewards).reshape(-1)[0])
                term = bool(np.asarray(terms).reshape(-1)[0])
                trunc = bool(np.asarray(truncs).reshape(-1)[0])
                succ = bool(np.asarray(env_infos["success"][0]).reshape(-1)[0])
                ep_actions.append(flat.astype(np.float32))
                ep_reward.append(r); ep_succ.append(succ)
                ep_success = ep_success or succ
                n_big += 1
                obs = next_obs
                if term or trunc:
                    break
            act_arr = np.stack(ep_actions, axis=0)
            np.save(out / f"episode_{ep:02d}_seed{seed}_actions.npy", act_arr)
            all_ms.extend(ep_ms)
            rec = {"episode": ep, "seed": seed, "success": bool(ep_success),
                   "n_bigsteps": n_big, "max_reward": float(np.max(ep_reward)),
                   "action_keys": keys_ref, "action_per_key_dims": dims_ref,
                   "action_total_dim": int(act_arr.shape[-1]),
                   "n_action_steps": int(act_arr.shape[1]),
                   "call0_obs_sha1": c0hash,
                   "server_ms_median": float(np.median(ep_ms)),
                   "actions_file": f"episode_{ep:02d}_seed{seed}_actions.npy"}
            episodes.append(rec)
            prog.write(json.dumps(rec) + "\n"); prog.flush()
            write_summary()  # incremental: robust to a later-episode failure
            print(f"[client] ep {ep} seed={seed} success={ep_success} bigsteps={n_big} "
                  f"max_reward={np.max(ep_reward):.3f} net_ms_med={np.median(ep_ms):.1f} "
                  f"c0={c0hash[:12]}", flush=True)
        except Exception as e:
            import traceback
            err = {"episode": ep, "seed": seed, "ERROR": repr(e)}
            prog.write(json.dumps(err) + "\n"); prog.flush()
            print(f"[client] ep {ep} seed={seed} FAILED: {e!r} -- continuing", flush=True)
            traceback.print_exc()
    env.close()
    prog.close()
    summary = write_summary()
    n_succ = summary["n_success"]
    print(f"[client] DONE success_rate={summary['success_rate']:.2f} "
          f"({n_succ}/{summary['n_completed']} completed of {args.n_episodes})", flush=True)


if __name__ == "__main__":
    main()
