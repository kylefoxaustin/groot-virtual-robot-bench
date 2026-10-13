#!/usr/bin/env python3
"""GR00T N1.5-3B torch-less HOST GLUE (numpy) for the iq9 board loop.

Reproduces, with numpy + the extracted constant tensors (optionB/glue_constants/),
the light host math that trt_model_forward.py / action_head_utils.py run as torch
between the on-NPU graphs. NO torch on the board. All float32 (matches numpy
promotion; the board graph IO is FLOAT_32, QNN casts to fp16 internally).

On-NPU graphs (executed by resident_orch, NOT here): vit, llm, vlln_vl_self_attention,
state_encoder, action_encoder, DiT, action_decoder.
Host glue (here): mlp1 projection, embed lookup + image-token scatter, action
position-embedding add, sa_embs concat, Euler flow-matching update.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, numpy as np, os

GC = os.environ.get("GROOT_GLUE", _W+"/optionB/glue_constants")

class GrootGlue:
    def __init__(self, glue_dir=GC):
        self.mlp1_w = np.load(f"{glue_dir}/mlp1_0_weight.npy").astype(np.float32)   # (2048,1152)
        self.mlp1_b = np.load(f"{glue_dir}/mlp1_0_bias.npy").astype(np.float32)     # (2048,)
        self.embed  = np.load(f"{glue_dir}/embed_tokens_weight.npy", mmap_mode="r") # (151680,2048) mmap
        self.future = np.load(f"{glue_dir}/ah_future_tokens_weight.npy").astype(np.float32)          # (32,1536)
        self.posemb = np.load(f"{glue_dir}/ah_position_embedding_weight.npy").astype(np.float32)     # (1024,1536)
        # config scalars (config.json action_head_cfg)
        self.action_horizon = 16
        self.action_dim = 32
        self.num_target_vision_tokens = 32
        self.add_pos_embed = True
        self.num_steps = 4              # num_inference_timesteps
        self.num_timestep_buckets = 1000
        self.dt = 1.0 / self.num_steps  # POSITIVE (GR00T flow t:0->1), Euler

    # --- VLM front-end glue ---
    def mlp1_project(self, vit_embeds):
        """(B, 256, 1152) siglip hidden -> (B, 256, 2048) LLM dim. mlp1 is a single Linear
        (checkpoint has ONLY mlp1.0.{weight,bias}; in=1152=siglip out => NO pixel_shuffle
        channel expansion for this checkpoint. VERIFY use_pixel_shuffle at board-prep)."""
        return vit_embeds.astype(np.float32) @ self.mlp1_w.T + self.mlp1_b

    def embed_and_scatter(self, input_ids, proj_vit_embeds, image_token_index):
        """input_ids (B,N) int -> input_embeds (B,N,2048), then replace rows where
        input_ids==image_token_index with the projected vit embeds (row-major order)."""
        B, N = input_ids.shape
        emb = np.asarray(self.embed[input_ids.reshape(-1)], dtype=np.float32).reshape(B, N, 2048)
        sel = (input_ids.reshape(-1) == image_token_index)
        flat = emb.reshape(B * N, 2048)
        flat[sel] = proj_vit_embeds.reshape(-1, 2048)[: sel.sum()]
        return flat.reshape(B, N, 2048)

    # --- action-head glue (per denoise step) ---
    def add_action_pos_embed(self, action_features):
        """action_features (B, 16, 1536) += position_embedding[0:16]."""
        if not self.add_pos_embed:
            return action_features
        n = action_features.shape[1]
        return action_features + self.posemb[:n][None].astype(np.float32)

    def assemble_sa_embs(self, state_features, action_features):
        """concat(state (B,1,1536), future_tokens (B,32,1536), action (B,16,1536), dim=1) -> (B,49,1536)."""
        B = state_features.shape[0]
        fut = np.broadcast_to(self.future[None], (B, *self.future.shape)).astype(np.float32)
        return np.concatenate([state_features, fut, action_features], axis=1).astype(np.float32)

    def euler(self, actions, pred):
        """actions += dt * pred[:, -action_horizon:]  (GR00T uses POSITIVE dt = 1/num_steps)."""
        vel = pred[:, -self.action_horizon:]
        return (actions + self.dt * vel).astype(np.float32)

    def t_discretized(self, step):
        return int((step / float(self.num_steps)) * self.num_timestep_buckets)  # 0,250,500,750


if __name__ == "__main__":
    # SMOKE TEST: prove the numpy glue is shape-correct against the extracted constants.
    g = GrootGlue()
    print("mlp1_w", g.mlp1_w.shape, "future", g.future.shape, "posemb", g.posemb.shape,
          "embed", g.embed.shape)
    ve = np.random.RandomState(0).randn(1, 256, 1152).astype(np.float32)
    proj = g.mlp1_project(ve); assert proj.shape == (1, 256, 2048), proj.shape
    ids = np.array([[0, 1, 2] + [999] * 256 + [3, 4]], dtype=np.int64)  # 999 = fake image tok
    ie = g.embed_and_scatter(ids, proj, 999); assert ie.shape == (1, 261, 2048), ie.shape
    sf = np.random.randn(1, 1, 1536).astype(np.float32)
    af = g.add_action_pos_embed(np.random.randn(1, 16, 1536).astype(np.float32))
    sa = g.assemble_sa_embs(sf, af); assert sa.shape == (1, 49, 1536), sa.shape
    a0 = np.zeros((1, 16, 32), np.float32)
    a1 = g.euler(a0, np.random.randn(1, 49, 32).astype(np.float32))
    assert a1.shape == (1, 16, 32)
    print("t_discretized 0..3 =", [g.t_discretized(s) for s in range(4)])
    print("SMOKE OK: glue shapes correct, dt=%.3f (positive)" % g.dt)
