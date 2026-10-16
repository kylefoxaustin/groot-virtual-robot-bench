"""The drone action head in numpy (float64), and the action-space gate metric.

Head: z-score(pooled GR00T feature 2048 ++ state 4) -> Linear 512 -> GELU -> LayerNorm -> Linear 512 -> GELU -> LayerNorm
-> Linear 64 -> (16, 4) -> de-normalise with the training-set action mean/std. 1,348,672 weights (drone/train/train_head.py
trains it; drone/assets/drone_head.npz is the one every published number used, normalisation statistics included).
"""
import numpy as np
from math import erf

DIMS = [0, 2, 3]          # vx, vz, yaw_rate (vy has zero variance in the data; the head emits ~0)
EXEC_STEPS = 4            # the gate scores the first 4 of 16 setpoints


def export(head_pt, feats_npz, out_npz):
    """torch checkpoint + the training features -> the numpy head file (stats recomputed exactly as training did:
    mean/std over ALL frames, torch's unbiased std, + 1e-6)."""
    import torch
    sd = torch.load(head_pt, map_location="cpu")["net"]
    z = np.load(feats_npz)
    f = torch.tensor(z["feats"]); s = torch.tensor(z["states"]); y = torch.tensor(z["chunks"])
    st = {"fmu": f.mean(0), "fsd": f.std(0) + 1e-6, "smu": s.mean(0), "ssd": s.std(0) + 1e-6,
          "ymu": y.reshape(-1, 4).mean(0), "ysd": y.reshape(-1, 4).std(0) + 1e-6}
    w = {k.replace(".", "_"): v.numpy() for k, v in sd.items()}
    np.savez(out_npz, **w, **{k: v.numpy() for k, v in st.items()})


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class Head:
    def __init__(self, npz):
        z = np.load(npz); self.p = {k: z[k].astype(np.float64) for k in z.files}
        try:
            from scipy.special import erf as verf
            self._erf = verf
        except ImportError:  # scipy is optional
            self._erf = np.vectorize(erf)

    def __call__(self, feat, state):
        p = self.p
        f = (np.atleast_2d(feat) - p["fmu"]) / p["fsd"]
        s = (np.atleast_2d(state) - p["smu"]) / p["ssd"]
        x = np.concatenate([f, s], -1)
        g = lambda t: 0.5 * t * (1.0 + self._erf(t / np.sqrt(2.0)))
        x = _ln(g(x @ p["net_0_weight"].T + p["net_0_bias"]), p["net_2_weight"], p["net_2_bias"])
        x = _ln(g(x @ p["net_3_weight"].T + p["net_3_bias"]), p["net_5_weight"], p["net_5_bias"])
        x = x @ p["net_6_weight"].T + p["net_6_bias"]
        a = x.reshape(-1, 16, 4)
        return a * p["ysd"] + p["ymu"]


def action_dev_sigma(a, ref, ysd):
    """Per-frame RMS deviation of the EXECUTED setpoints (first 4 steps; vx, vz, yaw_rate) in units of the dataset
    action std. Returns (per_frame_rms[N], pooled_rms)."""
    d = (a[:, :EXEC_STEPS, DIMS] - ref[:, :EXEC_STEPS, DIMS]) / ysd[DIMS]
    per = np.sqrt((d ** 2).mean(axis=(1, 2)))
    return per, float(np.sqrt((d ** 2).mean()))
