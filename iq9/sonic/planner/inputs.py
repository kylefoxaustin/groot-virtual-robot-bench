import numpy as np
DEF_Q = None
def make(mode=2, vel=-1.0, mv=(1,0,0), face=(1,0,0), seed=1234, height=-1.0, ctx=None):
    if ctx is None:
        ctx = np.zeros((1,4,36),np.float32); ctx[0,:,2]=0.78874; ctx[0,:,3]=1.0
    return {
     'context_mujoco_qpos': ctx.astype(np.float32),
     'target_vel': np.array([vel],np.float32),
     'mode': np.array([mode],np.int64),
     'movement_direction': np.array([mv],np.float32),
     'facing_direction': np.array([face],np.float32),
     'random_seed': np.array([seed],np.int64),
     'has_specific_target': np.zeros((1,1),np.int64),
     'specific_target_positions': np.zeros((1,4,3),np.float32),
     'specific_target_headings': np.zeros((1,4),np.float32),
     'allowed_pred_num_tokens': np.array([[0,0,0,1,1,1,0,0,0,0,0]],np.int64),
     'height': np.array([height],np.float32)}
