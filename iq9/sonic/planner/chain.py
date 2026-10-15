# Split-planner chain: host ORT stages H0..H4 interleaved with NPU blocks S,R,P,D.
# block_fn(name, feeds{float32 arrays}) -> {out_name: array}. Default = ORT fp32 on the block ONNX.
import onnxruntime as ort, numpy as np, json, os, time
D='blocks'   # relative to the planner work dir
ORDER=['S','R','P','D']
class Chain:
    def __init__(self, threads=4, block_fn=None, blocks_dir=D):
        so=ort.SessionOptions(); so.intra_op_num_threads=threads; so.inter_op_num_threads=1
        self.B=json.load(open(f'{blocks_dir}/blocks.json')); self.H=json.load(open(f'{blocks_dir}/host.json'))
        self.hs=[ort.InferenceSession(f'{blocks_dir}/host_H{k}.onnx',so,providers=['CPUExecutionProvider']) for k in range(5)]
        self.block_fn=block_fn
        if block_fn is None:
            self.bs={b:ort.InferenceSession(f'{blocks_dir}/planner_{b}.onnx',so,providers=['CPUExecutionProvider']) for b in ORDER}
    def block_feeds(self,b,env):
        f={}
        for x in self.B[b]['ins']:
            v=env[x]
            f[x if v.dtype==np.float32 else x+'__f32']=v.astype(np.float32)
        return f
    def run_block(self,b,feeds):
        if self.block_fn: return self.block_fn(b,feeds)
        s=self.bs[b]; return dict(zip([o.name for o in s.get_outputs()],s.run(None,feeds)))
    def __call__(self,inp,trace=None):
        env=dict(inp); t={}
        for k in range(5):
            h=self.hs[k]; names=[i.name for i in h.get_inputs()]
            t0=time.perf_counter(); r=h.run(None,{n:env[n] for n in names}); t[f'H{k}']=time.perf_counter()-t0
            env.update(zip([o.name for o in h.get_outputs()],r))
            if k<4:
                b=ORDER[k]; f=self.block_feeds(b,env)
                if trace is not None: trace[b+'_in']=f
                t0=time.perf_counter(); o=self.run_block(b,f); t[b]=time.perf_counter()-t0
                if trace is not None: trace[b+'_out']=o
                env.update(o)
        return env['mujoco_qpos'],env['num_pred_frames'],t
