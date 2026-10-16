#!/usr/bin/env python3
"""Export the SAME seeded seg stand-in (drone/contention/make_standins.py) with its dilated convolutions rewritten into
mathematically identical dense ops, for the IQ-9075 NPU (a dilated conv whose dilation is large relative to its feature map
was very slow on v73 HTP in our build; this rewrite removes every dilation and keeps the output):
zero-Pad by the conv padding, take one strided slice per kernel tap (offset = tap index x dilation), and
  - dense conv (groups 1):  Concat the tap slices on channels -> one 1x1 Conv with the tap-major reshaped weight
  - depthwise conv:         sum over taps of slice x per-channel weight (Mul + Add)
Taps whose slice lies ENTIRELY in the zero padding contribute exactly 0 and are dropped (exact, not an approximation:
at 512 px the ASPP map is 32 x 32, so the dilation-36 conv keeps only its centre tap).
Weights are the seeded stand-in's (make_standins.py procedure, rebuilt here from the same seed); at 512 the output is
checked against the canonical seg_512.onnx in fp32 (onnxruntime) on the seeded check tiles before anything is written
(make_seg1024_check.py does the same at 1024).
usage: python3 drone/iq9/contention/make_dilfree.py [OUT_DIR] [--which aspp|all] [--res 512|1024] [--standins DIR]
       (default OUT_DIR = --standins = $DRONE_STANDINS, which make_standins.py filled). Writes seg_<res>_dilfree.onnx + .json.
Needs torch, torchvision, onnx, onnxruntime, onnx_graphsurgeon."""
import sys, os, json, hashlib, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default=None); ap.add_argument("--which", default="all", choices=["aspp", "all"])
ap.add_argument("--res", type=int, default=512); ap.add_argument("--standins", default=None); a = ap.parse_args()
A = a.standins or os.environ.get("DRONE_STANDINS") or sys.exit("set --standins or source drone/env.sh")
a.out = a.out or A
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "contention", "make_standins.py")).read()
assert "# --- seeded check inputs" in src and "os.makedirs(OUT, exist_ok=True)" in src, "make_standins.py changed: update the rebuild below"
src = src.split("# --- seeded check inputs")[0].replace("os.makedirs(OUT, exist_ok=True)", "").replace("torch.onnx.export", "(lambda *x, **k: None)")
sys_argv = sys.argv; sys.argv = [sys.argv[0], "/dev/null"]; g = {"__name__": "x"}; exec(compile(src, "make_standins_models", "exec"), g); sys.argv = sys_argv
seg = g["seg"]; SEED = g["SEED"]


class DilFreeConv(nn.Module):
    def __init__(self, conv: nn.Conv2d, H: int, W: int):
        super().__init__()
        assert conv.stride == (1, 1) and conv.bias is None and conv.padding_mode == "zeros"
        self.p, self.d, k = conv.padding, conv.dilation, conv.kernel_size
        self.dw = conv.groups == conv.in_channels == conv.out_channels and conv.groups > 1
        assert self.dw or conv.groups == 1
        self.H, self.W = H, W
        taps = []
        for ky in range(k[0]):
            for kx in range(k[1]):
                oy, ox = ky * self.d[0], kx * self.d[1]          # slice start in the padded map
                # rows oy..oy+H-1 of the padded map; real data occupies p..p+H-1. Fully in padding -> exact zero.
                if oy + H <= self.p[0] or oy >= self.p[0] + H or ox + W <= self.p[1] or ox >= self.p[1] + W: continue
                taps.append((ky, kx, oy, ox))
        self.taps = taps
        w = conv.weight.detach()
        if self.dw:
            self.register_buffer("wt", torch.stack([w[:, 0, ky, kx].view(1, -1, 1, 1) for ky, kx, _, _ in taps]))
        else:
            self.register_buffer("wt", torch.cat([w[:, :, ky, kx] for ky, kx, _, _ in taps], 1)[:, :, None, None])   # [O, T*C, 1, 1]

    def forward(self, x):
        xp = F.pad(x, (self.p[1], self.p[1], self.p[0], self.p[0]))
        sl = [xp[:, :, oy:oy + self.H, ox:ox + self.W] for _, _, oy, ox in self.taps]
        if self.dw:
            y = sl[0] * self.wt[0]
            for t in range(1, len(sl)): y = y + sl[t] * self.wt[t]
            return y
        return F.conv2d(torch.cat(sl, 1) if len(sl) > 1 else sl[0], self.wt)


# shapes of every dilated conv input at this resolution
shapes = {}
hooks = [m.register_forward_hook(lambda mod, i, o, n=n: shapes.__setitem__(n, tuple(i[0].shape)))
         for n, m in seg.named_modules() if isinstance(m, nn.Conv2d) and m.dilation != (1, 1)]
with torch.no_grad(): seg(torch.zeros(1, 3, a.res, a.res))
for h in hooks: h.remove()
rep = {}
for n, shp in shapes.items():
    if a.which == "aspp" and not n.startswith("m.classifier"): continue
    parent = seg.get_submodule(n.rsplit(".", 1)[0]); leaf = n.rsplit(".", 1)[1]
    conv = getattr(parent, leaf); new = DilFreeConv(conv, shp[2], shp[3]); setattr(parent, leaf, new)
    rep[n] = {"dilation": list(conv.dilation), "kernel": list(conv.kernel_size), "groups": conv.groups, "in": shp, "taps_kept": len(new.taps),
              "taps_total": conv.kernel_size[0] * conv.kernel_size[1], "form": "depthwise Mul+Add" if new.dw else "Concat + 1x1 Conv"}
os.makedirs(a.out, exist_ok=True)
name = f"seg_{a.res}_dilfree{'' if a.which == 'all' else '_aspp'}"
p = f"{a.out}/{name}.onnx"
with torch.no_grad():
    torch.onnx.export(seg, torch.zeros(1, 3, a.res, a.res), p, input_names=["img"], output_names=["seg"], opset_version=17,
                      dynamo=False, training=torch.onnx.TrainingMode.EVAL, do_constant_folding=True)
import onnxruntime as ort, onnx, collections
import onnx_graphsurgeon as gs
# fold the shape-only subgraph torch emits for F.pad (ConstantOfShape/Reshape/Transpose/Cast) into constant pads
G_ = gs.import_onnx(onnx.load(p)); G_.fold_constants().cleanup().toposort(); onnx.save(gs.export_onnx(G_), p)
I = np.load(f"{A}/check_inputs_fp32.npz"); R = np.load(f"{A}/check_ref_fp32.npz")
s_new = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
outs = np.stack([s_new.run(None, {"img": t})[0] for t in I["seg_tiles"]]) if a.res == 512 else None
chk = {}
if outs is not None:
    ref = R["seg_out"]
    chk = {"max_abs_vs_canonical_onnx_fp32": float(np.abs(outs - ref).max()), "ref_absmax": float(np.abs(ref).max()),
           "argmax_agreement_vs_canonical": float((outs.argmax(2) == ref.argmax(2)).mean())}
    assert chk["max_abs_vs_canonical_onnx_fp32"] < 1e-3, chk
ops = collections.Counter(nd.op_type for nd in onnx.load(p).graph.node)
dil = [att.ints for nd in onnx.load(p).graph.node if nd.op_type == "Conv" for att in nd.attribute if att.name == "dilations" and list(att.ints) != [1, 1]]
meta = {"onnx": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "seed": SEED, "res": a.res, "which": a.which,
        "rewritten": rep, "op_counts": dict(ops), "dilated_convs_left": len(dil), "fp32_equivalence": chk}
json.dump(meta, open(f"{a.out}/{name}.json", "w"), indent=1); print(json.dumps(meta, indent=1))
