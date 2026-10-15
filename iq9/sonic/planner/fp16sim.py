# fp16 emulation (for predict.py; run in the planner work dir: python3 fp16sim.py -> blocks_sim/) of an NPU block ONNX by graph rewrite (host ORT):
#  f16store: every weight, block input and every float intermediate rounded to fp16 (fp32 compute in between)
#  f16acc  : f16store + MatMul/Gemm/Conv reductions split into K-chunks of ~32 MACs, running sum rounded to fp16
import onnx, numpy as np, pickle, sys, math
from onnx import helper, numpy_helper, TensorProto
VALS=None
def _vals():
    global VALS
    if VALS is None: VALS=pickle.load(open('vals_walk.pkl','rb'))
    return VALS
def transform(src, dst, mode, chunk=32):
    m=onnx.load(src); g=m.graph; V=_vals()
    shape={}; dtype={}
    for k,v in V.items(): shape[k]=v.shape; dtype[k]=v.dtype
    for i in g.input:
        shape[i.name]=tuple(d.dim_value for d in i.type.tensor_type.shape.dim); dtype[i.name]=np.dtype('float32')
    inits={}
    for t in g.initializer:
        a=numpy_helper.to_array(t)
        if a.dtype==np.float32: a=a.astype(np.float16).astype(np.float32)
        inits[t.name]=a; shape[t.name]=a.shape; dtype[t.name]=a.dtype
    newnodes=[]; uid=[0]
    def nm(s): uid[0]+=1; return f'{s}__{uid[0]}'
    def rnd(x_raw, x):  # fp16 round-trip x_raw -> x
        h=nm(x+'_h'); newnodes.append(helper.make_node('Cast',[x_raw],[h],to=TensorProto.FLOAT16,name=nm('c16')))
        newnodes.append(helper.make_node('Cast',[h],[x],to=TensorProto.FLOAT,name=nm('c32')))
    def const(a,base):
        n=nm(base); inits[n]=np.asarray(a); return n
    # round graph inputs
    ren={}
    for i in g.input:
        if i.type.tensor_type.elem_type==TensorProto.FLOAT:
            r=i.name+'__r16'; rnd(i.name,r); ren[i.name]=r
    for n in g.node:
        ins=[ren.get(x,x) for x in n.input]
        if n.op_type=='Constant':
            a=numpy_helper.to_array(n.attribute[0].t) if n.attribute[0].name=='value' else None
            if a is not None and a.dtype==np.float32:
                inits[n.output[0]]=a.astype(np.float16).astype(np.float32); continue
            newnodes.append(n); continue
        outs=list(n.output)
        isf=[o and dtype.get(o)==np.float32 for o in outs]
        raw=[o+'__raw' if f else o for o,f in zip(outs,isf)]
        done=False
        if mode=='f16acc' and n.op_type in ('MatMul','Gemm','Conv'):
            done=chunked(n,ins,raw[0],inits,shape,newnodes,nm,rnd,const,chunk)
        if not done:
            nn=helper.make_node(n.op_type,ins,raw,name=n.name); nn.attribute.extend(n.attribute); newnodes.append(nn)
        for o,r,f in zip(outs,raw,isf):
            if f: rnd(r,o)
    for k,nd in enumerate(newnodes):
        if not nd.name: nd.name=f'sim_n{k}'
    del g.node[:]; g.node.extend(newnodes)
    del g.initializer[:]; g.initializer.extend([numpy_helper.from_array(a,k) for k,a in inits.items()])
    onnx.save(m,dst,save_as_external_data=True,all_tensors_to_one_file=True,location=dst.split('/')[-1]+'.data',size_threshold=1024)
def chunked(n,ins,out,inits,shape,newnodes,nm,rnd,const,chunk):
    attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
    if n.op_type=='Gemm':
        A,W=ins[0],ins[1]; C=ins[2] if len(ins)>2 else None
        if W not in inits or attrs.get('transA',0): return False
        w=inits[W]; w=w.T if attrs.get('transB',0) else w
        K=w.shape[0]; parts=[]
        acc=None
        for k0 in range(0,K,chunk):
            a_s=nm('as'); newnodes.append(helper.make_node('Slice',[A,const(np.array([k0]),'st'),const(np.array([min(K,k0+chunk)]),'en'),const(np.array([-1]),'ax')],[a_s]))
            p=nm('p'); newnodes.append(helper.make_node('MatMul',[a_s,const(w[k0:k0+chunk],'w')],[p]))
            acc=accum(acc,p,newnodes,nm,rnd)
        if C: s=nm('b'); newnodes.append(helper.make_node('Add',[acc,C],[s])); acc=s
        newnodes.append(helper.make_node('Identity',[acc],[out])); return True
    if n.op_type=='MatMul':
        A,B=ins; sa=shape.get(n.input[0]); K=sa[-1] if sa else None
        if K is None or K<=chunk: return False
        acc=None
        for k0 in range(0,K,chunk):
            k1=min(K,k0+chunk)
            a_s=nm('as'); newnodes.append(helper.make_node('Slice',[A,const(np.array([k0]),'st'),const(np.array([k1]),'en'),const(np.array([-1]),'ax')],[a_s]))
            if B in inits: b_s=const(inits[B][...,k0:k1,:],'w')
            else:
                b_s=nm('bs'); newnodes.append(helper.make_node('Slice',[B,const(np.array([k0]),'st'),const(np.array([k1]),'en'),const(np.array([-2]),'ax')],[b_s]))
            p=nm('p'); newnodes.append(helper.make_node('MatMul',[a_s,b_s],[p]))
            acc=accum(acc,p,newnodes,nm,rnd)
        newnodes.append(helper.make_node('Identity',[acc],[out])); return True
    if n.op_type=='Conv':
        X,W=ins[0],ins[1]; B=ins[2] if len(ins)>2 else None
        if W not in inits or attrs.get('group',1)!=1: return False
        w=inits[W]; cin=w.shape[1]; ksz=int(np.prod(w.shape[2:])); cc=max(1,chunk//ksz)
        if cin<=cc: return False
        acc=None
        kw={k:v for k,v in attrs.items()}
        for c0 in range(0,cin,cc):
            c1=min(cin,c0+cc)
            x_s=nm('xs'); newnodes.append(helper.make_node('Slice',[X,const(np.array([c0]),'st'),const(np.array([c1]),'en'),const(np.array([1]),'ax')],[x_s]))
            p=nm('p'); newnodes.append(helper.make_node('Conv',[x_s,const(w[:,c0:c1],'w')],[p],**kw))
            acc=accum(acc,p,newnodes,nm,rnd)
        if B:
            bb=inits[B].reshape([1,-1]+[1]*(len(w.shape)-2)); s=nm('b'); newnodes.append(helper.make_node('Add',[acc,const(bb,'bias')],[s])); acc=s
        newnodes.append(helper.make_node('Identity',[acc],[out])); return True
    return False
def accum(acc,p,newnodes,nm,rnd):
    if acc is None:
        r=nm('r'); rnd(p,r); return r
    s=nm('s'); newnodes.append(helper.make_node('Add',[acc,p],[s])); r=nm('r'); rnd(s,r); return r
if __name__=='__main__':
    import os
    os.makedirs('blocks_sim',exist_ok=True)
    for b in 'SRPD':
        for mode in ('f16store','f16acc'):
            transform(f'blocks/planner_{b}.onnx',f'blocks_sim/planner_{b}_{mode}.onnx',mode); print(b,mode,'ok',flush=True)
