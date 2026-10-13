#!/usr/bin/env python3
"""Bake embodiment_id=EID into an AH graph: replace every GatherND(weight_table, eid) with a
STATIC initializer = table[eid], and drop the GatherND ops + the embodiment_id input + its
index-prep nodes. Converts activation x DYNAMIC-gathered-weight MatMuls (which some QNN versions'
HTP fp16 path can mishandle) into standard activation x STATIC-weight MatMuls. EID is fixed per deployment (gr1=24).
Usage: bake_eid.py <in.onnx> <out.onnx>"""
import sys, numpy as np, onnx
from onnx import numpy_helper, helper
EID=24
inp, outp = sys.argv[1], sys.argv[2]
m=onnx.load(inp, load_external_data=True); g=m.graph
inits={i.name:numpy_helper.to_array(i) for i in g.initializer}

# find GatherND nodes; their input[0] is a weight table initializer, input[1] the eid index tensor
gnd=[n for n in g.node if n.op_type=="GatherND"]
new_static={}   # gatherND output name -> baked array (1,...)
idx_tensors=set()
for n in gnd:
    tbl=n.input[0]; idx_tensors.add(n.input[1])
    assert tbl in inits, f"table {tbl} not initializer"
    arr=inits[tbl]
    baked=arr[EID:EID+1]        # keep leading dim=1 (matches GatherND (1,...) output)
    new_static[n.output[0]]=np.ascontiguousarray(baked)

# drop GatherND nodes; drop nodes that only produce the eid index (trace back from idx_tensors to embodiment_id)
drop_names={n.name for n in gnd}
# index-prep nodes: any node whose output is in idx_tensors AND consumes embodiment_id (Unsqueeze)
for n in g.node:
    if n.output and n.output[0] in idx_tensors and ("embodiment_id" in n.input):
        drop_names.add(n.name)
keep=[n for n in g.node if n.name not in drop_names]
del g.node[:]; g.node.extend(keep)

# add baked initializers
for name,arr in new_static.items():
    g.initializer.append(numpy_helper.from_array(arr.astype(arr.dtype), name=name))

# remove embodiment_id input
new_in=[i for i in g.input if i.name!="embodiment_id"]
del g.input[:]; g.input.extend(new_in)
def pin(vi):
    d=vi.type.tensor_type.shape.dim[0]; d.ClearField("dim_param"); d.dim_value=1
for i in g.input: pin(i)
for o in g.output: pin(o)
# prune unused initializers (old tables + index consts)
used=set(); [used.update(n.input) for n in g.node]
keep_i=[ini for ini in g.initializer if ini.name in used]; del g.initializer[:]; g.initializer.extend(keep_i)
onnx.checker.check_model(m, full_check=False)
onnx.save(m, outp, save_as_external_data=True, location=outp.split("/")[-1]+".data", all_tensors_to_one_file=True)
print(f"BAKED {inp} -> {outp}")
print("  gatherND removed:", len(gnd), "| inputs now:", [(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input], "| #nodes", len(g.node))
