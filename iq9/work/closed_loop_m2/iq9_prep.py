#!/usr/bin/env python3
"""closed_loop_m2: obs -> iq9 GFRAME raws. Reproduces GR00T's OWN preprocessing (the same
Gr00tPolicy.apply_transforms + backbone/action_head.prepare_input path used to
build the board's step-0 raws in groot_capture_ref.py), so the iq9 sees the SAME
transformed observation a GPU GR00T server sees.

Board raw contract (verified by dumpGraphIO on the 7 baked v73 fp16 ctx bins):
  groot_vit:   pixel_values FLOAT_32 (1,3,224,224)=150528 ; position_ids INT_32 256
  groot_llm:   inputs_embeds (built on-board) ; attention_mask INT_32 N=296
  vlln/DiT:    N=296 sequence (hard-baked) -> input_ids MUST be exactly 296 tokens
  state_encoder: state FLOAT_32 64
  action loop: init_actions FLOAT_32 (16,32)=512 ; time_proj_t{0,250,500,750} (fixed)

INT_32 inputs (position_ids/attention_mask) are read on-board via setInputFromF32File
which reads FLOAT32 values and static_casts to int32 -> so we write them as float32 VALUES.
input_ids.raw is read as int64 (8 bytes/token).

N=296 reconciliation: RoboCasa's instruction "unlocked_waist: pick up the milk, ..."
tokenizes to N=302; the "unlocked_waist: " wrapper prefix is exactly 6 tokens, and
dropping it yields EXACTLY N=296 with the full task instruction intact (lossless for
the milk task, all 5 seeds identical). A general force_to_296 safety net handles any
other length by trimming/padding the instruction-text region (logged per call).
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, json, numpy as np

IMAGE_TOKEN = 151669   # from capture_config.json (verified from the model)
N_BAKED = 296
WAIST_PREFIX = "unlocked_waist: "


def build_transform(model_path, data_config_name="fourier_gr1_arms_waist"):
    """Build the modality transform with metadata set (no 3B model load needed)."""
    from gr00t.experiment.data_config import load_data_config
    from gr00t.data.schema import DatasetMetadata
    dc = load_data_config(data_config_name)
    tr = dc.transform(); tr.eval()
    md = json.load(open(f"{model_path}/experiment_cfg/metadata.json"))["gr1"]
    tr.set_metadata(DatasetMetadata.model_validate(md))
    return tr, dc


def _find_image_block(ids):
    idx = np.where(ids == IMAGE_TOKEN)[0]
    return int(idx[0]), int(idx[-1])


def force_to_296(input_ids, attention_mask):
    """Return (ids296 int64, mask296 float32, info). Strip nothing here (prefix already
    stripped upstream); if len!=296 adjust the instruction-text region right before the
    trailing assistant-prompt tokens, preserving the 256-image block + chat structure."""
    ids = np.asarray(input_ids).reshape(-1).astype(np.int64)
    am = np.asarray(attention_mask).reshape(-1).astype(np.int64)
    n = ids.shape[0]
    info = {"N_in": int(n), "action": "none"}
    if n == N_BAKED:
        return ids, am.astype(np.float32), info
    i0, i1 = _find_image_block(ids)
    # trailing template tail = tokens after the instruction (from last <|im_end|>=151645)
    tail_start = int(np.where(ids == 151645)[0][-1]) if (ids == 151645).any() else n - 5
    head = ids[: i1 + 2]          # pre + image block + </img>(i1+1)
    instr = ids[i1 + 2 : tail_start]
    tail = ids[tail_start:]
    target_instr = N_BAKED - head.shape[0] - tail.shape[0]
    if target_instr < 0:
        # extreme: truncate tail too (should never happen for this task)
        ids2 = np.concatenate([head, tail])[:N_BAKED]
        info["action"] = "hard_truncate"
    elif instr.shape[0] > target_instr:
        instr2 = instr[:target_instr]      # drop END of instruction text
        ids2 = np.concatenate([head, instr2, tail])
        info["action"] = f"truncate_instr_{instr.shape[0]-target_instr}"
    else:
        pad = np.full(target_instr - instr.shape[0], instr[-1] if instr.size else 198, np.int64)
        ids2 = np.concatenate([head, instr, pad, tail])
        info["action"] = f"pad_instr_{target_instr-instr.shape[0]}"
    assert ids2.shape[0] == N_BAKED, ids2.shape
    mask2 = np.ones(N_BAKED, np.float32)   # all real (no left-padding introduced)
    info["N_out"] = N_BAKED
    return ids2.astype(np.int64), mask2, info


class GrootPrep:
    """Holds the transform; converts a raw obs dict (as GR00T's simulation client sends) into the
    board raws. Uses the model's backbone/action_head prepare_input via a light shim."""

    def __init__(self, model_path, data_config_name="fourier_gr1_arms_waist"):
        self.tr, self.dc = build_transform(model_path, data_config_name)
        self.model_path = model_path

    def strip_prefix(self, obs):
        k = "annotation.human.coarse_action"
        if k in obs:
            v = obs[k]
            def strip1(s):
                s = s.decode() if isinstance(s, bytes) else str(s)
                return s[len(WAIST_PREFIX):] if s.startswith(WAIST_PREFIX) else s
            if isinstance(v, np.ndarray):
                obs[k] = np.array([strip1(x) for x in v.reshape(-1)]).reshape(v.shape)
            elif isinstance(v, (list, tuple)):
                obs[k] = [strip1(x) for x in v]
            else:
                obs[k] = strip1(v)
        return obs

    def unapply(self, action_norm):
        """action_norm: (1,16,32) normalized action_pred from the board -> unnormalized
        action dict {action.left_arm (1,16,7), ...} exactly as Gr00tPolicy._get_unnormalized_action
        (is_batch=True path, no squeeze). Uses the SAME transform.unapply + metadata as Gr00tPolicy."""
        import torch
        t = torch.as_tensor(np.asarray(action_norm), dtype=torch.float32).reshape(1, 16, 32)
        return self.tr.unapply({"action": t.cpu()})

    def obs_to_raws(self, obs):
        """obs: dict as GR00T's simulation client sends (batched B=1,T=1). Returns dict of numpy arrays
        keyed by board raw name + an 'info' dict. Mirrors groot_capture_ref exactly."""
        from gr00t.model.policy import unsqueeze_dict_values
        obs = {k: v for k, v in obs.items()}
        # client-side key fix (verbatim from SimulationInferenceClient.get_action):
        # the served obs key is video.ego_view; raw env obs uses the *_bg_crop_pad_* key.
        if "video.ego_view" not in obs and "video.ego_view_bg_crop_pad_res256_freq20" in obs:
            obs["video.ego_view"] = obs.pop("video.ego_view_bg_crop_pad_res256_freq20")
        obs = self.strip_prefix(obs)
        # match Gr00tPolicy.get_action batching contract
        is_batch = np.asarray(obs["state.left_arm"]).ndim == 3
        if not is_batch:
            obs = unsqueeze_dict_values(obs)
        for k, v in obs.items():
            if not isinstance(v, np.ndarray):
                obs[k] = np.array(v)
        norm = self.tr(obs)
        pixel = np.asarray(norm["eagle_pixel_values"]).astype(np.float32)      # (1,3,224,224)
        ids = np.asarray(norm["eagle_input_ids"]).astype(np.int64)
        am = np.asarray(norm["eagle_attention_mask"]).astype(np.int64)
        state = np.asarray(norm["state"]).astype(np.float32)                   # (1,1,64)
        ids296, mask296, tinfo = force_to_296(ids, am)
        position_ids = np.arange(256, dtype=np.float32)                        # written as f32 -> int32
        return {
            "pixel_values": np.ascontiguousarray(pixel),
            "input_ids": np.ascontiguousarray(ids296),                         # int64
            "attention_mask": np.ascontiguousarray(mask296),                   # f32 -> int32
            "position_ids": np.ascontiguousarray(position_ids),                # f32 -> int32
            "state": np.ascontiguousarray(state.reshape(-1)),                  # 64 f32
        }, tinfo


if __name__ == "__main__":
    # offline self-test on saved reset obs (no board, no 3B model)
    import pickle, sys
    OBS = sys.argv[1] if len(sys.argv) > 1 else \
        _gvrb_os.path.join(_W, "closed_loop_m2", "probe_obs")
    p = GrootPrep(_gvrb_os.environ.get("GROOT_MODEL", _W+"/models/GR00T-N1.5-3B"))
    for ep in range(5):
        obs = pickle.load(open(f"{OBS}/reset_seed{ep}.pkl", "rb"))
        raws, info = p.obs_to_raws(obs)
        print(f"seed{ep}: N_info={info} pixel={raws['pixel_values'].shape} "
              f"ids={raws['input_ids'].shape} mask_sum={int(raws['attention_mask'].sum())} "
              f"state={raws['state'].shape} state[:4]={raws['state'][:4].round(3)}")
