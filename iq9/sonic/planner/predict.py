# fp16 predictor for the split planner on the logged planner calls of a reference GPU run (x86 host only, BEFORE the board run).
# usage (planner work dir, after fp16sim.py): python3 predict.py <planner refset.npz from sonic/gate/make_refsets.py> pred.json
# Writes pred.json (fp32 / f16store / f16acc / planted arms; the board gate's bracket) and pred.json.<arm>.npz.
import numpy as np, sys, os, json, time, onnxruntime as ort
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from inputs import make; from chain import Chain, ORDER; from pmetrics import metrics
ref=np.load(sys.argv[1]); out=sys.argv[2]
N=len(ref['mode'])
def inputs(i):
    return make(int(ref['mode'][i]),float(ref['target_vel'][i]),ref['movement_direction'][i],ref['facing_direction'][i],int(ref['seed'][i]),float(ref['height'][i]),ctx=ref['context_mujoco_qpos'][i][None])
def blockfn(mode, plant=False):
    so=ort.SessionOptions(); so.intra_op_num_threads=16
    ss={b:ort.InferenceSession(f'blocks_sim/planner_{b}_{mode}.onnx' if mode!='fp32' else f'blocks/planner_{b}.onnx',so,providers=['CPUExecutionProvider'],disabled_optimizers=['GatherSliceToSplitFusion']) for b in ORDER}
    def f(b,feeds):
        if plant and b=='P': feeds={k:(v[...,::-1].copy() if k=='/Add_135_output_0' else v) for k,v in feeds.items()}
        s=ss[b]; return dict(zip([o.name for o in s.get_outputs()],s.run(None,feeds)))
    return f
res={}
ref_codes=None
for arm,mode,plant in [('fp32','fp32',False),('f16store','f16store',False),('f16acc','f16acc',False),('plant_f16store','f16store',True)]:
    c=Chain(threads=16,block_fn=blockfn(mode,plant))
    Q=[];Nf=[];C=[]
    t0=time.time()
    for i in range(N):
        tr={}; q,n,_=c(inputs(i),trace=tr)
        Q.append(q[0]); Nf.append(int(n[0])); C.append(tr['P_out']['/mmm_net/_proj_pose_output_logit/Add_output_0'].reshape(16,8,10).argmax(-1))
    Q=np.array(Q);Nf=np.array(Nf);C=np.array(C)
    if arm=='fp32': ref_codes=C
    m=metrics(ref['ort_qpos'],ref['ort_npf'],Q,Nf,ref_codes,C); m['secs']=time.time()-t0
    res[arm]=m; print(arm,json.dumps(m),flush=True)
    np.savez(f'{out}.{arm}.npz',qpos=Q,npf=Nf,codes=C)
# the reference GPU run's TensorRT planner vs ORT fp32, same metrics, for scale
res['trt_ref_vs_fp32']=metrics(ref['ort_qpos'],ref['ort_npf'],ref['trt_qpos'],ref['trt_npf'])
json.dump(res,open(out,'w'),indent=1)
