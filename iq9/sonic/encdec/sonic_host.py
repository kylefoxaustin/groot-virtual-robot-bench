"""Host-side FSQ glue for the IQ-9075 SONIC encoder split: the NPU graph stops before the FSQ Round; the CPU rounds and selects."""
import numpy as np
def fsq_select(pre_round, obs):
    """pre_round [N,3,2,32] float, obs [N,1762] -> encoded_tokens [N,64] (upstream forward value)."""
    tok = np.round(pre_round.astype(np.float32)) / np.float32(16.0)   # STE: s+(round(s)-s) == round(s)
    k = obs[:, 0].astype(np.int64)
    return tok[np.arange(len(k)), k].reshape(len(k), 64).astype(np.float32)
