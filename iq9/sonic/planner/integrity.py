# Pipeline integrity: re-run the host stages ON THE HOST, feeding the board's own NPU block outputs (dumped), and
# compare (a) the block inputs the host computes vs the inputs the board fed the NPU, (b) final qpos/npf.
# usage (planner work dir): python3 integrity.py <board run dir made with --dump_blocks> <planner refset.npz>
import numpy as np, json, sys, os, re, onnxruntime as ort
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from inputs import make; from chain import Chain
d=sys.argv[1]
ref=np.load(sys.argv[2]); N=len(ref['mode'])
san=lambda s: re.sub('[^A-Za-z0-9_]','_',s)
lay={}
for k,b in enumerate('SRPD'):
    info=json.load(open(f'qnn/ctx/planner_{b}_aaf0_fp16.info.json'))['info']['graphs'][0]['info']
    ins=[(t['info']['name'],t['info']['dimensions']) for t in info['graphInputs']]; outs=[(t['info']['name'],t['info']['dimensions']) for t in info['graphOutputs']]
    a=np.fromfile(f'{d}/blk{k}_in.f32',np.float32).reshape(N,-1); o=np.fromfile(f'{d}/blk{k}_out.f32',np.float32).reshape(N,-1)
    lay[b]=(ins,outs,a,o)
cur=[0]; worst={b:0.0 for b in 'SRPD'}
def bf(b,feeds):
    ins,outs,a,o=lay[b]; i=cur[0]; off=0
    for qn,sh in ins:
        sz=int(np.prod(sh)); v=a[i,off:off+sz]; off+=sz
        k=[x for x in feeds if san(x)==qn][0]; worst[b]=max(worst[b],float(np.abs(feeds[k].ravel()-v).max()))
    res={}; off=0
    B=json.load(open('blocks/blocks.json'))[b]
    for qn,sh in outs:
        sz=int(np.prod(sh)); nm=[x for x in B['outs'] if san(x)==qn][0]; res[nm]=o[i,off:off+sz].reshape(sh); off+=sz
    return res
c=Chain(block_fn=bf); q=np.fromfile(f'{d}/qpos.f32',np.float32).reshape(N,64,36); nb=np.fromfile(f'{d}/npf.i32',np.int32)
dq=0; dn=0
for i in range(N):
    cur[0]=i
    inp=make(int(ref['mode'][i]),float(ref['target_vel'][i]),ref['movement_direction'][i],ref['facing_direction'][i],int(ref['seed'][i]),float(ref['height'][i]),ctx=ref['context_mujoco_qpos'][i][None])
    qq,nn,_=c(inp); dq=max(dq,float(np.abs(qq[0,:nn[0]]-q[i,:nn[0]]).max())); dn+=int(nn[0]!=nb[i])
print('max |host-recomputed block input - board block input| per block:',worst); print('final qpos max diff (host stages on host vs on board, same NPU outputs):',dq,'npf mismatches',dn)
