"""fp32-output onnxruntime-CPU reference for one stage, from the SAME stage-input files the engines are fed.
usage: python3 profiling/compiled_fleet/mkref.py {vit|llm|dit}
Reads  $GVRB_STAGE_IO (default $GVRB_WORK/trtwork/io; written by dump_stage_inputs.py) and $GROOT_ONNX (default $GVRB_WORK/onnx)
Writes $GVRB_WORK/trtwork/ref_<stage>.npy (what thor_gate.py / orin gate.py / trt_groot.py gate against).
Optional cross-check: if a previously stored reference $GVRB_WORK/qnn/io/g<stage>_ref.npy exists, its cosine and
rel-MAE against the fresh reference are printed (results/profiling/groot_compiled_fleet_2026-08-27.json records
that the stored DiT reference differed from a fresh ORT run)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import numpy as np, onnxruntime as ort, sys, os, json
IO=_gvrb_os.environ.get("GVRB_STAGE_IO", _W+'/trtwork/io')
OUT=_W+'/trtwork'
ONX=_gvrb_os.environ.get('GROOT_ONNX', _W+'/onnx')
def r(f,dt,shape):
    p=os.path.join(IO,f)
    if not os.path.exists(p): sys.exit(f"missing {p} -- run profiling/compiled_fleet/dump_stage_inputs.py first")
    a=np.fromfile(p,dtype=dt); return a.reshape(shape)
stage=sys.argv[1]
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if stage=='vit':
    p=ONX+'/eagle2/vit_fp16.onnx'
    feeds={'pixel_values': r('vit_pixel_values.raw',np.float16,(1,3,224,224)),
           'position_ids': r('vit_position_ids.raw',np.int64,(1,256))}
elif stage=='llm':
    p=ONX+'/eagle2/llm_fp16.onnx'
    feeds={'inputs_embeds': r('llm_inputs_embeds.raw',np.float16,(1,296,2048)),
           'attention_mask': r('llm_attention_mask.raw',np.int64,(1,296))}
else:
    p=ONX+'/action_head/DiT_fp16.onnx'
    feeds={'sa_embs': r('dit_sa_embs.raw',np.float16,(1,49,1536)),
           'vl_embs': r('dit_vl_embs.raw',np.float16,(1,296,2048)),
           'timesteps_tensor': r('dit_timesteps_tensor.raw',np.int64,(1,))}
s=ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
print('inputs',[(i.name,i.type,i.shape) for i in s.get_inputs()])
o=s.run(None,feeds)[0].astype(np.float32)
os.makedirs(OUT,exist_ok=True)
np.save(f'{OUT}/ref_{stage}.npy',o)
print(stage,o.shape,o.dtype,'absmean=%.6g'%np.abs(o).mean(),'min=%.4g max=%.4g'%(o.min(),o.max()))
# optional cross-check vs a previously stored reference
q=os.path.join(_W,'qnn','io',{'vit':'gvit_ref.npy','llm':'gllm_ref.npy','dit':'gdit_ref.npy'}[stage])
if os.path.exists(q):
    qq=np.load(q).astype(np.float32).reshape(o.shape)
    a=o.ravel().astype(np.float64); b=qq.ravel().astype(np.float64)
    cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    rel=float(np.abs(a-b).mean()/np.abs(b).mean())
    print('CROSSCHECK vs %s: cosine=%.6f rel_mae=%.6f'%(q,cos,rel))
