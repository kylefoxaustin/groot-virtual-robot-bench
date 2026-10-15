# Planner gate metrics: arm outputs vs the host ORT fp32 reference (full upstream ONNX) on the logged calls.
import numpy as np
def quat_ang_deg(q1,q2):
    d=np.abs(np.sum(q1*q2,-1))/ (np.linalg.norm(q1,axis=-1)*np.linalg.norm(q2,axis=-1)+1e-12)
    return np.degrees(2*np.arccos(np.clip(d,0,1)))
def metrics(ref_q,ref_n,q,n,ref_codes=None,codes=None):
    N=len(ref_n); match=(n==ref_n)
    rp=[];jt=[];qa=[];rpmax=[];jmax=[]
    for i in range(N):
        if not match[i]: continue
        k=int(ref_n[i]); a=q[i,:k]; b=ref_q[i,:k]
        e=np.linalg.norm(a[:,0:3]-b[:,0:3],axis=-1); rp.append(e); rpmax.append(e.max())
        j=np.abs(a[:,7:36]-b[:,7:36]); jt.append(j.ravel()); jmax.append(j.max())
        qa.append(quat_ang_deg(a[:,3:7],b[:,3:7]))
    rp=np.concatenate(rp) if rp else np.zeros(1); jt=np.concatenate(jt) if jt else np.zeros(1); qa=np.concatenate(qa) if qa else np.zeros(1)
    out=dict(n_calls=int(N),npf_agree=float(match.mean()),npf_mismatch_calls=[int(i) for i in np.nonzero(~match)[0]],
      root_pos_err_m=dict(max=float(rp.max()),p99=float(np.percentile(rp,99)),mean=float(rp.mean())),
      joint_err_rad=dict(max=float(jt.max()),p99=float(np.percentile(jt,99)),mean=float(jt.mean())),
      root_quat_err_deg=dict(max=float(qa.max()),p99=float(np.percentile(qa,99))),
      calls_with_joint_err_gt_0p1rad=int(sum(m>0.1 for m in jmax)))
    if codes is not None:
        same=(codes==ref_codes); out['pose_code_agree']=float(same.mean()); out['calls_any_code_flip']=float((~same.reshape(N,-1)).any(1).mean())
    return out
