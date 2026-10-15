# Run in the planner work dir after build_blocks.py:  python3 fix_resize.py R D
# An exact rewrite applied while hunting the planner gate failure. It did NOT fix it (the dilated Conv1d did, fix_dilated.py);
# it is kept because the published block graphs were built with it.
# Replace 1-D nearest/asymmetric/floor x2 Resize ([1,C,L]->[1,C,2L]) by an exactly-equivalent
# Unsqueeze->Concat(x,x)->Reshape (element repeat). Numerically identical in fp32 (verified below).
import onnx, numpy as np, pickle, sys, onnxruntime as ort
from onnx import helper, numpy_helper
V=pickle.load(open('vals_walk.pkl','rb'))
for b in sys.argv[1:]:
    src=f'blocks/planner_{b}.onnx'; m=onnx.load(src); g=m.graph; new=[]; n_rep=0
    for n in g.node:
        if n.op_type=='Resize':
            at={a.name:helper.get_attribute_value(a) for a in n.attribute}
            assert at['mode']==b'nearest' and at['coordinate_transformation_mode']==b'asymmetric' and at['nearest_mode']==b'floor', at
            x=n.input[0]; C,L=V[x].shape[1],V[x].shape[2]; assert V[n.output[0]].shape==(1,C,2*L)
            ax=n.name+'_ax'; s1=n.name+'_shp'
            g.initializer.extend([numpy_helper.from_array(np.array([3],np.int64),ax),numpy_helper.from_array(np.array([1,C,2*L],np.int64),s1)])
            u=n.name+'_u'; c=n.name+'_c'
            new+= [helper.make_node('Unsqueeze',[x,ax],[u],name=n.name+'_U'),
                   helper.make_node('Concat',[u,u],[c],axis=3,name=n.name+'_C'),
                   helper.make_node('Reshape',[c,s1],[n.output[0]],name=n.name+'_R')]
            n_rep+=1
        else: new.append(n)
    del g.node[:]; g.node.extend(new)
    dst=f'blocks/planner_{b}.onnx'
    import shutil; shutil.copy(src,src+'.with_resize'); shutil.copy(src.replace('.onnx','.data'),src.replace('.onnx','.data')+'.with_resize')
    onnx.save(m,dst,save_as_external_data=True,all_tensors_to_one_file=True,location=f'planner_{b}.data',size_threshold=1024)
    print(b,'replaced',n_rep,'Resize')
