# usage (planner work dir): python3 blkcheck.py <board run dir made with --dump_blocks> [n_calls]
# Per-block check of a board dump: board block output vs host ORT fp32 of the same block on the board's own block inputs.
import numpy as np, json, sys, re, onnxruntime as ort
d=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else len(np.fromfile(f'{d}/npf.i32',np.int32))
B=json.load(open('blocks/blocks.json'))
san=lambda s: re.sub('[^A-Za-z0-9_]','_',s)
for k,b in enumerate('SRPD'):
    info=json.load(open(f'qnn/ctx/planner_{b}_aaf0_fp16.info.json'))['info']['graphs'][0]['info']
    ins=[(t['info']['name'],t['info']['dimensions']) for t in info['graphInputs']]
    outs=[(t['info']['name'],t['info']['dimensions']) for t in info['graphOutputs']]
    a=np.fromfile(f'{d}/blk{k}_in.f32',np.float32); o=np.fromfile(f'{d}/blk{k}_out.f32',np.float32)
    isz=[int(np.prod(s)) for _,s in ins]; osz=[int(np.prod(s)) for _,s in outs]
    a=a.reshape(N,sum(isz)); o=o.reshape(N,sum(osz))
    s=ort.InferenceSession(f'blocks/planner_{b}.onnx',providers=['CPUExecutionProvider'])
    onames=[x.name for x in s.get_outputs()]
    errs={n:[] for n,_ in outs}
    for i in range(N):
        feeds={}; off=0
        for (qn,sh),sz in zip(ins,isz):
            on=[x.name for x in s.get_inputs() if san(x.name)==qn][0]
            feeds[on]=a[i,off:off+sz].reshape(sh); off+=sz
        r=dict(zip(onames,s.run(None,feeds)))
        off=0
        for (qn,sh),sz in zip(outs,osz):
            on=[x for x in onames if san(x)==qn][0]
            ref=r[on].ravel(); got=o[i,off:off+sz]; off+=sz
            errs[qn].append((np.abs(got-ref).mean()/ (np.abs(ref).mean()+1e-12), np.abs(got-ref).max()))
    for qn,e in errs.items():
        e=np.array(e); print(b,qn,'rel_mae mean %.3g max %.3g | maxabs max %.3g'%(e[:,0].mean(),e[:,0].max(),e[:,1].max()))
