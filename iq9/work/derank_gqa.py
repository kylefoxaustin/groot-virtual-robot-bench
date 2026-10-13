#!/usr/bin/env python3
"""Rewrite GR00T's grouped-query-attention repeat_kv from rank-5 to rank-4 for QNN.

QNN's HTP validator rejects the exported GQA pattern:
    Unsqueeze(x[B,H,S,D], axis=2) -> [B,H,1,S,D]
    Expand(...)                   -> [B,H,R,S,D]      <-- RANK 5, refused
    Reshape(...)                  -> [B,H*R,S,D]
That chain is exactly repeat_interleave along the head axis by R, and it can be written
at rank 4 instead:
    Reshape(x -> [B*H, 1, S, D])
    Concat([x,x,...R times], axis=1) -> [B*H, R, S, D]
    Reshape(-> [B, H*R, S, D])
Head ordering is preserved: element (b,h,r) lands at head h*R+r, i.e. h0,h0,h1,h1,...
which is what repeat_interleave produces — NOT what a plain concat of the 4-D tensor
would produce (h0,h1,...,h0,h1,...). Getting that backwards would be a silent
wrong-answer, so the rewrite is verified numerically against the original graph before
it is used.
"""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

def dims(vi):
    return [d.dim_value if d.dim_value else (d.dim_param or "?")
            for d in vi.type.tensor_type.shape.dim]

def main(src, dst):
    m = onnx.load(src)                      # WITH external data so we can save a complete model
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
    def shp(nm):
        return dims(vi[nm]) if nm in vi else None

    new_nodes, drop, added = [], set(), 0
    for n in g.node:
        if n.op_type != "Expand":
            continue
        out_s = shp(n.output[0])
        in_s = shp(n.input[0])
        if not out_s or len(out_s) != 5 or not in_s or len(in_s) != 5:
            continue
        un = prod.get(n.input[0])
        if un is None or un.op_type != "Unsqueeze":
            continue
        rs = [c for c in cons.get(n.output[0], []) if c.op_type == "Reshape"]
        if len(rs) != 1:
            continue
        rs = rs[0]
        x = un.input[0]                       # [B,H,S,D]
        xs = shp(x)
        if not xs or len(xs) != 4:
            continue
        B, H, S, D = xs
        R = out_s[2]
        if not isinstance(R, int) or R < 2:
            continue
        pre = f"{n.name}_derank"
        sh1 = numpy_helper.from_array(np.array([-1, 1, 0, 0], dtype=np.int64), pre + "_s1")
        # ⚠️ batch MUST be -1 (inferred), not 0. In a Reshape with allowzero=0 a 0 means
        # "copy the corresponding INPUT dim" — and the input here is [B*H, R, S, D], so a 0
        # in slot 0 copies B*H (=8), not B (=1). That produced a shape of {8,16,296,128}
        # for a tensor of 606208 elements and onnxruntime rejected it. Caught by the
        # equivalence check, which is the only reason it did not become a silent rewrite.
        # Take the head count from the EXPAND'S OUTPUT, not the input. One of these chains
        # has a symbolic head dim on the input side ('(8//s43)') while the output carries the
        # concrete 8 and 2 — using the input gave a second -1 and onnxruntime refused the
        # graph ("Target shape may not have multiple -1 dimensions").
        hr = out_s[1] * out_s[2] if isinstance(out_s[1], int) and isinstance(out_s[2], int) else None
        if hr is None:
            continue
        sh2 = numpy_helper.from_array(np.array([-1, hr, 0, 0], dtype=np.int64), pre + "_s2")
        g.initializer.extend([sh1, sh2])
        r1 = helper.make_node("Reshape", [x, sh1.name], [pre + "_r1"], name=pre + "_r1n", allowzero=0)
        cc = helper.make_node("Concat", [pre + "_r1"] * R, [pre + "_cc"], name=pre + "_ccn", axis=1)
        r2 = helper.make_node("Reshape", [pre + "_cc", sh2.name], [rs.output[0]],
                              name=pre + "_r2n", allowzero=0)
        new_nodes.extend([(n.name, [r1, cc, r2])])
        drop.update({un.name, n.name, rs.name})
        added += 1

    if not added:
        sys.exit("  ✗ no rank-5 GQA expand pattern found — nothing rewritten")
    repl = {k: v for k, v in new_nodes}
    out_nodes = []
    for n in g.node:
        if n.name in repl:
            out_nodes.extend(repl[n.name]); continue
        if n.name in drop:
            continue
        out_nodes.append(n)
    del g.node[:]
    g.node.extend(out_nodes)
    onnx.save(m, dst, save_as_external_data=True,
              location=dst.split("/")[-1] + ".data", all_tensors_to_one_file=True)
    print(f"  rewrote {added} rank-5 GQA expand chain(s) to rank-4 Reshape/Concat/Reshape")
    print(f"  wrote {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
