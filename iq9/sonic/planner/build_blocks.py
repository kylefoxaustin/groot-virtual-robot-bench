# Build the four per-block NPU subgraphs (S,R,P,D) from the upstream planner ONNX (run in the planner work dir, after dump_vals.py).
# Each block is the closure of its primary output (spec.json) back to the last data-dependent/host op; tensors that do not depend on the
# block inputs are constant-folded from vals_walk.pkl; int/bool block inputs become float32 inputs with an in-graph Cast (QNN-friendly I/O).
# The host-stage graphs H0..H4 between the blocks are built by build_host.py.  usage: python3 build_blocks.py
import onnx, json, pickle, numpy as np, sys, os, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from inputs import make
OUT='blocks'; os.makedirs(OUT,exist_ok=True)
m=onnx.load('planner_sonic.onnx'); g=m.graph
prod={}; cons={}
for i,n in enumerate(g.node):
    for o in n.output: prod[o]=i
    for x in n.input: cons.setdefault(x,[]).append(i)
inits={i.name:i for i in g.initializer}
HOST={'ArgMax','ArgMin','TopK','OneHot','CumSum','ScatterND','ScatterElements','GatherElements','Mod','Range','IsNaN','Floor','Einsum'}
SPEC=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'spec.json')))
isC=lambda n:n.op_type=='Constant'
# full-run values for folding (input-independent tensors only are used)
vals=pickle.load(open('vals_walk.pkl','rb'))
T2NP={1:np.float32,6:np.int32,7:np.int64,9:np.bool_}
blocks={}
for b,(lo,prim) in SPEC.items():
    blk=set(); ins=[]; st=list(prim); seen=set()
    while st:
        t=st.pop()
        if t in seen: continue
        seen.add(t)
        if t in inits: continue
        if t not in prod: ins.append(t); continue
        p=prod[t]; n=g.node[p]
        if isC(n): blk.add(p); continue
        allconst=all((x in inits) or (x in prod and isC(g.node[prod[x]])) for x in n.input if x)
        if (p<lo or n.op_type in HOST) and not allconst: ins.append(t); continue
        blk.add(p); st+=[x for x in n.input if x]
    ins=sorted(set(ins)); dyn=set(ins)
    for i in sorted(blk):
        n=g.node[i]
        if n.op_type in ('Shape',): continue
        if any(x in dyn for x in n.input): dyn.update(n.output)
    S=set(blk)
    dnodes=[i for i in sorted(blk) if any(o in dyn for o in g.node[i].output)]
    fold=sorted({x for i in dnodes for x in g.node[i].input if x and x not in dyn and x not in inits and not isC(g.node[prod[x]])})
    cnodes=sorted({prod[x] for i in dnodes for x in g.node[i].input if x and x in prod and isC(g.node[prod[x]])})
    outs=[t for i in dnodes for t in g.node[i].output if (t in prim or any(c not in S for c in cons.get(t,[]))) and vals[t].dtype==np.float32]
    # graph
    new_inits=[]; nodes=[]; ginputs=[]
    for x in fold:
        new_inits.append(numpy_helper.from_array(np.asarray(vals[x]),x))
    used_inits={x for i in dnodes for x in g.node[i].input if x in inits}
    new_inits+=[inits[x] for x in sorted(used_inits)]
    for x in ins:
        v=vals[x]
        if v.dtype==np.float32:
            ginputs.append(helper.make_tensor_value_info(x,TensorProto.FLOAT,list(v.shape)))
        else:
            fx=x+'__f32'
            ginputs.append(helper.make_tensor_value_info(fx,TensorProto.FLOAT,list(v.shape)))
            to={np.dtype('bool'):TensorProto.BOOL,np.dtype('int64'):TensorProto.INT64,np.dtype('int32'):TensorProto.INT32}[v.dtype]
            nodes.append(helper.make_node('Cast',[fx],[x],to=to,name=x+'__cast'))
    nodes+=[g.node[i] for i in cnodes]+[g.node[i] for i in dnodes]
    goutputs=[helper.make_tensor_value_info(t,TensorProto.FLOAT,list(vals[t].shape)) for t in outs]
    gg=helper.make_graph(nodes,f'planner_{b}',ginputs,goutputs,new_inits)
    mm=helper.make_model(gg,opset_imports=m.opset_import); mm.ir_version=m.ir_version
    fn=f'{OUT}/planner_{b}.onnx'
    onnx.save(mm,fn,save_as_external_data=True,all_tensors_to_one_file=True,location=f'planner_{b}.data',size_threshold=1024)
    nparam=sum(int(np.prod(t.dims)) for t in new_inits if t.name in used_inits)
    blocks[b]=dict(ins=ins,in_dtypes={x:str(vals[x].dtype) for x in ins},outs=outs,n_nodes=len(dnodes),params=nparam,
                   ops=dict(Counter(g.node[i].op_type for i in dnodes).most_common()))
    print(b,'nodes',len(dnodes),'params %.1fM'%(nparam/1e6),'ins',[(x,str(vals[x].dtype),vals[x].shape) for x in ins],'outs',[(t,vals[t].shape) for t in outs])
    print('   ops',blocks[b]['ops'])
json.dump(blocks,open(f'{OUT}/blocks.json','w'),indent=1)
