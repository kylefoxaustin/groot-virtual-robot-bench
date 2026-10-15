# Host (CPU, onnxruntime fp32) stage graphs H0..H4 between the NPU blocks; closure-extracted from the original graph.
# They hold every ArgMax/TopK/OneHot/CumSum/ScatterND and the seed hash, which stay off the NPU. Run after build_blocks.py.
import onnx, json, numpy as np, os
from onnx import utils, helper
from collections import Counter
m=onnx.load('planner_sonic.onnx')
B=json.load(open('blocks/blocks.json'))
order=['S','R','P','D']
gin=[i.name for i in m.graph.input]
avail=list(gin); stages={}; produced={}
ex=utils.Extractor(m); subs={}
for k,b in enumerate(order+['END']):
    outs=B[b]['ins'] if b!='END' else ['mujoco_qpos','num_pred_frames']
    sub=ex.extract_model(avail,outs)
    # drop unused inputs
    used={x for n in sub.graph.node for x in n.input}|set(outs)
    keep=[i for i in sub.graph.input if i.name in used]
    del sub.graph.input[:]; sub.graph.input.extend(keep)
    ops=Counter(n.op_type for n in sub.graph.node)
    heavy={o:ops[o] for o in ('MatMul','Conv','Gemm','LayerNormalization','Softmax') if o in ops}
    # carry forward: tensors produced here are available to later host stages
    produced[k]=[o for n in sub.graph.node if n.op_type!='Constant' for o in n.output if o and o not in ('mujoco_qpos','num_pred_frames')]
    fn=f'blocks/host_H{k}.onnx'
    subs[k]=sub
    onnx.save(sub,fn,save_as_external_data=True,all_tensors_to_one_file=True,location=f'host_H{k}.data',size_threshold=1<<20)
    print(f'H{k} -> {b}: nodes {len(sub.graph.node)} in {[i.name for i in keep]} out {outs} heavy {heavy}')
    stages[f'H{k}']=dict(inputs=[i.name for i in keep],outputs=outs,nodes=len(sub.graph.node),heavy=heavy)
    if b!='END': avail+=B[b]['outs']+produced[k]
for k in range(len(order)+1):
    need=set(stages[f'H{k}']['inputs'])
    for j in range(k):
        extra=[t for t in produced[j] if t in need and t not in stages[f'H{j}']['outputs']]
        if extra:
            vi=[helper.make_empty_tensor_value_info(t) for t in extra]
            subs[j].graph.output.extend(vi); stages[f'H{j}']['outputs']+=extra
for k,sub in subs.items():
    onnx.save(sub,f'blocks/host_H{k}.onnx',save_as_external_data=True,all_tensors_to_one_file=True,location=f'host_H{k}.data',size_threshold=1<<20)
    print(f'H{k}: nodes {len(sub.graph.node)} n_in {len(stages[f"H{k}"]["inputs"])} n_out {len(stages[f"H{k}"]["outputs"])}')
json.dump(stages,open('blocks/host.json','w'),indent=1)
