# Run in the planner work dir after build_blocks.py (+ fix_resize.py):  python3 fix_dilated.py R D
# Why: the Hexagon v73 HTP backend computed the dilated Conv1d layers of blocks R and D (dilation 3 and 9 on short sequences)
# silently wrong in fp16 -- no error, plausible-looking output -- and the planner gate failed. The rewrite below is exact in fp32.
# Replace every dilated 1-D Conv (k=3, dilation d, pads d) in a block by an exactly-equivalent form QNN/HTP gets right:
#   d >= L : only the centre tap ever sees data -> Conv k=1 with the centre weight
#   d <  L : Pad(d) -> 3 shifted Slices -> Concat(channels) -> Conv k=1 with weights [W0|W1|W2]
import onnx, numpy as np, pickle, sys, shutil
from onnx import helper, numpy_helper
V=pickle.load(open('vals_walk.pkl','rb'))
for b in sys.argv[1:]:
    src=f'blocks/planner_{b}.onnx'; m=onnx.load(src); g=m.graph
    inits={t.name:t for t in g.initializer}; new=[]; cnt={'center':0,'shift':0}
    for n in g.node:
        at={a.name:helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type=='Conv' and at.get('dilations',[1])[0]>1:
            d=at['dilations'][0]; assert at['kernel_shape']==[3] and at['pads']==[d,d] and at.get('group',1)==1 and at.get('strides',[1])==[1]
            x=n.input[0]; W=numpy_helper.to_array(inits[n.input[1]]); B=n.input[2] if len(n.input)>2 else None
            C,L=V[x].shape[1],V[x].shape[2]
            if d>=L:
                wn=n.name+'_wc'; g.initializer.append(numpy_helper.from_array(W[:,:,1:2].copy(),wn))
                new.append(helper.make_node('Conv',[x,wn]+([B] if B else []),list(n.output),name=n.name+'_center',kernel_shape=[1],pads=[0,0],dilations=[1],strides=[1],group=1))
                cnt['center']+=1
            else:
                pn=n.name+'_pads'; g.initializer.append(numpy_helper.from_array(np.array([0,0,d,0,0,d],np.int64),pn))
                xp=n.name+'_xp'; new.append(helper.make_node('Pad',[x,pn],[xp],name=n.name+'_pad',mode='constant'))
                sl=[]
                for j in range(3):
                    s0=n.name+f'_s{j}0'; s1=n.name+f'_s{j}1'; ax=n.name+f'_ax{j}'
                    g.initializer.extend([numpy_helper.from_array(np.array([j*d],np.int64),s0),numpy_helper.from_array(np.array([j*d+L],np.int64),s1),numpy_helper.from_array(np.array([2],np.int64),ax)])
                    o=n.name+f'_sl{j}'; new.append(helper.make_node('Slice',[xp,s0,s1,ax],[o],name=n.name+f'_slice{j}')); sl.append(o)
                cc=n.name+'_cat'; new.append(helper.make_node('Concat',sl,[cc],axis=1,name=n.name+'_concat'))
                wn=n.name+'_w3'; g.initializer.append(numpy_helper.from_array(np.concatenate([W[:,:,0:1],W[:,:,1:2],W[:,:,2:3]],axis=1),wn))
                new.append(helper.make_node('Conv',[cc,wn]+([B] if B else []),list(n.output),name=n.name+'_k1',kernel_shape=[1],pads=[0,0],dilations=[1],strides=[1],group=1))
                cnt['shift']+=1
        else: new.append(n)
    del g.node[:]; g.node.extend(new)
    shutil.copy(src,src+'.pre_dil'); shutil.copy(src.replace('.onnx','.data'),src.replace('.onnx','.data')+'.pre_dil')
    onnx.save(m,src,save_as_external_data=True,all_tensors_to_one_file=True,location=f'planner_{b}.data',size_threshold=1024)
    print(b,cnt)
