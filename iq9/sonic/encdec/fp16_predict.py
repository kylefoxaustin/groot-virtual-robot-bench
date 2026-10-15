#!/usr/bin/env python3
"""fp16 predictor: what SHOULD a correct fp16 Hexagon build of the SONIC encoder/decoder score
on these exact inputs?  Run on the x86 host BEFORE any board measurement, so the board result is judged
against a bracket written in advance.

Usage:  python -I fp16_predict.py <ref.npz> <onnx_dir> <out_pred.json> [--n N]
ref.npz keys: enc_in [N,1762], enc_out [N,64], dec_in [N,994], dec_out [N,29] (float32): the encdec reference set from
sonic/gate/make_refsets.py (the tensors a reference GPU run consumed and produced). <onnx_dir> = the output of make_graphs.py.

Arms (all computed from the SAME inputs, by a small numpy ONNX interpreter of our two graphs):
  fp32      : float32 everywhere (sanity: must match onnxruntime fp32 to ~1e-6)
  f16store  : every weight and every intermediate tensor rounded to fp16, matmul accumulated in fp32
              (= what HTP fp16_relaxed_precision nominally does: fp16 storage/compute, wide accumulate)
  f16acc    : f16store + the matmul running sum rounded to fp16 every 32 MACs (pessimistic bracket)
Reference for every metric = onnxruntime fp32 on the host (NOT the TRT log, whose own precision is
reported separately as trt_vs_fp32).
Also writes <out_pred>_ref.npz (ref_tok, ref_act) for gate.py and loop_check.py.

Encoder metrics are on encoded_tokens (after Round/16 and mode select, same glue the board path uses):
  token_exact_frac  = fraction of the N*64 token values bit-equal to the fp32 reference
  tick_anyflip_rate = fraction of ticks with >=1 of the 64 tokens different
Decoder metrics on action (radians, joint-position targets):  rel_mae = mean|a-ref|/mean|ref|,
  max_abs_rad, cos.  Decoder is fed the LOGGED dec_in (so its error is isolated from encoder flips);
  a 'coupled' decoder metric (decoder fed with the arm's own tokens) is also reported if
  dec_in[:, :64] == enc_out (verified, not assumed).
"""
import sys, json, time, hashlib, argparse
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sonic_host import fsq_select

F16 = np.float16


def q(x, mode):
    if mode == "fp32" or x.dtype != np.float32:
        return x
    return x.astype(F16).astype(np.float32)


def matmul(a, b, mode):
    if mode != "f16acc":
        return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    K = a.shape[-1]
    acc = np.zeros(a.shape[:-1] + (b.shape[-1],), np.float32)
    for k0 in range(0, K, 32):
        part = (a[..., k0:k0 + 32].astype(np.float64) @ b[k0:k0 + 32].astype(np.float64)).astype(np.float32)
        acc = (acc + part).astype(F16).astype(np.float32)
    return acc


class Interp:
    def __init__(self, path):
        m = onnx.load(path)
        self.g = m.graph
        self.inits = {i.name: numpy_helper.to_array(i) for i in self.g.initializer}

    def run(self, feeds, mode):
        env = {}
        for k, v in self.inits.items():
            env[k] = q(v.astype(np.float32), mode) if v.dtype == np.float32 else v
        for k, v in feeds.items():
            env[k] = q(v, mode)
        for n in self.g.node:
            I = [env[x] if x else None for x in n.input]
            A = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            t = n.op_type
            if t == "Constant":
                o = numpy_helper.to_array(A["value"])
            elif t == "Slice":
                data, st, en = I[0], I[1], I[2]
                ax = I[3] if len(I) > 3 and I[3] is not None else np.arange(len(st))
                stp = I[4] if len(I) > 4 and I[4] is not None else np.ones(len(st), np.int64)
                sl = [slice(None)] * data.ndim
                for s, e, a_, p in zip(st, en, ax, stp):
                    sl[int(a_)] = slice(int(s), int(min(e, 2**62)), int(p))
                o = data[tuple(sl)]
            elif t == "Unsqueeze":
                o = I[0]
                for a_ in sorted(int(x) % (o.ndim + 1) for x in I[1]):
                    o = np.expand_dims(o, a_)
            elif t == "Squeeze":
                o = np.squeeze(I[0], axis=tuple(int(x) for x in I[1]))
            elif t == "Reshape":
                shp = [int(s) for s in I[1]]
                shp = [I[0].shape[i] if s == 0 else s for i, s in enumerate(shp)]
                o = I[0].reshape(shp)
            elif t == "Concat":
                o = np.concatenate(I, axis=A["axis"])
            elif t == "Gemm":
                b = I[1].T if A.get("transB", 0) else I[1]
                o = matmul(I[0], b, mode) * A.get("alpha", 1.0)
                o = q(o, mode) + I[2] * A.get("beta", 1.0)
            elif t == "MatMul":
                o = matmul(I[0], I[1], mode)
            elif t == "Add":
                o = I[0] + I[1]
            elif t == "Sub":
                o = I[0] - I[1]
            elif t == "Mul":
                o = I[0] * I[1]
            elif t == "Div":
                o = I[0] / I[1]
            elif t == "Sigmoid":
                o = (1.0 / (1.0 + np.exp(-I[0].astype(np.float64)))).astype(np.float32)
            elif t == "Tanh":
                o = np.tanh(I[0]).astype(np.float32)
            else:
                raise NotImplementedError(t)
            if isinstance(o, np.ndarray) and o.dtype == np.float64:
                o = o.astype(np.float32)
            env[n.output[0]] = q(o, mode) if isinstance(o, np.ndarray) else o
        return [env[o.name] for o in self.g.output]


def dec_metrics(a, ref):
    d = np.abs(a - ref)
    cos = float(np.sum(a * ref) / (np.linalg.norm(a) * np.linalg.norm(ref) + 1e-30))
    return {"rel_mae": float(d.mean() / np.abs(ref).mean()), "max_abs_rad": float(d.max()),
            "p99_abs_rad": float(np.percentile(d, 99)), "cos": cos}


def enc_metrics(tok, ref):
    eq = tok == ref
    return {"token_exact_frac": float(eq.mean()), "tick_anyflip_rate": float((~eq.all(1)).mean()),
            "n_token_flips": int((~eq).sum()), "max_flip_levels": float(np.abs(tok - ref).max() * 16)}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("onnx_dir"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=0, help="limit ticks (0=all)")
    ap.add_argument("--label", default="REAL")
    a = ap.parse_args()
    z = np.load(a.npz)
    N = len(z["enc_in"]) if a.n == 0 else min(a.n, len(z["enc_in"]))
    enc_in, dec_in = z["enc_in"][:N].astype(np.float32), z["dec_in"][:N].astype(np.float32)
    t0 = time.time()
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    full = ort.InferenceSession(f"{a.onnx_dir}/sonic_enc_full.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{a.onnx_dir}/sonic_dec.onnx", so, providers=["CPUExecutionProvider"])
    ref_tok = np.concatenate([full.run(None, {"obs_dict": enc_in[i:i + 1]})[0] for i in range(N)])
    ref_act = np.concatenate([dec.run(None, {"obs_dict": dec_in[i:i + 1]})[0] for i in range(N)])
    ec, dc = Interp(f"{a.onnx_dir}/sonic_enc_core.onnx"), Interp(f"{a.onnx_dir}/sonic_dec.onnx")
    tok_slot = bool(np.array_equal(dec_in[:, :64], z["enc_out"][:N]))
    out = {"label": a.label, "N": N, "npz": a.npz,
           "input_sha256_16": {"enc_in": sha(enc_in), "dec_in": sha(dec_in)},
           "ref": "onnxruntime fp32 CPU, graph opts disabled",
           "trt_vs_fp32": {"enc": enc_metrics(z["enc_out"][:N], ref_tok),
                           "dec": dec_metrics(z["dec_out"][:N].astype(np.float32), ref_act)},
           "dec_in_first64_equals_enc_out": tok_slot, "arms": {}}
    # pre-Round margin: how close is each value to a rounding boundary (k+0.5)?
    pre = np.stack([ec.run({"obs_dict": enc_in[i:i + 1]}, "fp32")[0] for i in range(N)])
    k = enc_in[:, 0].astype(np.int64)
    sel = pre[np.arange(N), k].reshape(N, 64)
    margin = np.abs(sel - (np.floor(sel) + 0.5))
    out["pre_round_margin"] = {"min": float(margin.min()), "p0.1": float(np.percentile(margin, 0.1)),
                               "p1": float(np.percentile(margin, 1)), "median": float(np.median(margin)),
                               "frac_below_0.01": float((margin < 0.01).mean()),
                               "frac_below_0.002": float((margin < 0.002).mean()),
                               "encoder_mode_counts": {int(m): int((k == m).sum()) for m in np.unique(k)}}
    for mode in ["fp32", "f16store", "f16acc"]:
        pr = np.stack([ec.run({"obs_dict": enc_in[i:i + 1]}, mode)[0] for i in range(N)])
        tok = fsq_select(pr, enc_in)
        act = np.concatenate([dc.run({"obs_dict": dec_in[i:i + 1]}, mode)[0] for i in range(N)])
        arm = {"enc": enc_metrics(tok, ref_tok), "dec": dec_metrics(act, ref_act)}
        if tok_slot:
            di = dec_in.copy(); di[:, :64] = tok
            act_c = np.concatenate([dc.run({"obs_dict": di[i:i + 1]}, mode)[0] for i in range(N)])
            arm["dec_coupled"] = dec_metrics(act_c, ref_act)
        out["arms"][mode] = arm
        print(mode, json.dumps(arm), flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    out["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    np.savez(a.out.replace(".json", "_ref.npz"), ref_tok=ref_tok, ref_act=ref_act)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
