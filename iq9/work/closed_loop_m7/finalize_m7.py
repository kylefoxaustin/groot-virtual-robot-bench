#!/usr/bin/env python3
"""closed_loop_m7 finalize: int8-vs-fp16 GR00T precision ladder on iq9 Hexagon v73, closed-loop
RoboCasa, apples-to-apples with closed_loop_m5. Per arm: success rate + Wilson CI, agreement vs the
5090 bf16 reference (per-seed flips), arm-joint fidelity distribution vs 5090, and
per-graph int8 latency vs fp16 (from the resident_orch per-graph timers). Embeds the
per-graph int8-vs-fp16 accuracy gate (whether int8 engaged + per-graph error)."""
import argparse, json, math, os
import numpy as np

def rel_mae(a,b,lo=None,hi=None):
    a=a.astype(np.float64); b=b.astype(np.float64)
    if lo is not None: a=a[...,lo:hi]; b=b[...,lo:hi]
    return float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=(z/d)*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return (round(p,4),round(max(0,c-h),4),round(min(1,c+h),4))
def summ(d): return json.load(open(os.path.join(d,"rollout_summary.json")))
def succ(s): return {e["seed"]:bool(e["success"]) for e in s["episodes"]}
def ep_idx(s): return {e["seed"]:e["episode"] for e in s["episodes"]}
def load_ep(d,ep,seed): return np.load(os.path.join(d,f"episode_{ep:02d}_seed{seed}_actions.npy"))

def calllog_lat(p):
    if not p or not os.path.exists(p): return None
    cl=json.load(open(p))
    def med(k):
        v=[c[k] for c in cl if c.get(k) is not None]; return round(float(np.median(v)),1) if v else None
    out={"n_calls":len(cl),"board_wall_ms_median":med("board_wall_ms"),
         "board_gframe_ms_median":med("board_gframe_ms")}
    for g in ["vit","llm","vlln","state","aenc","dit","adec_sum"]:
        vv=[c[g] for c in cl if c.get(g) is not None]
        if vv: out[f"{g}_us_median"]=round(float(np.median(vv)),1)
    return out

def arm_metrics(arm_dir, ref_dir, calllog):
    s_arm=summ(arm_dir); s_ref=summ(ref_dir); ma=succ(s_arm); mr=succ(s_ref)
    seeds=sorted(set(ma)&set(mr)); N=len(seeds)
    ka=sum(ma[s] for s in seeds); kr=sum(mr[s] for s in seeds)
    agree=[s for s in seeds if ma[s]==mr[s]]; dis=[s for s in seeds if ma[s]!=mr[s]]
    eia=ep_idx(s_arm); eir=ep_idx(s_ref); per={}
    pa=[]; pb=[]
    for s in seeds:
        a=load_ep(arm_dir,eia[s],s); b=load_ep(ref_dir,eir[s],s); n=min(a.shape[0],b.shape[0]); a,b=a[:n],b[:n]
        per[str(s)]=round(rel_mae(a,b,0,14),5); pa.append(a[...,0:14].reshape(-1)); pb.append(b[...,0:14].reshape(-1))
    pooled=rel_mae(np.concatenate(pa),np.concatenate(pb)); vals=list(per.values())
    return {
      "N":N,"success":{"k":ka,"n":N,"rate":wilson(ka,N)[0],"wilson95":list(wilson(ka,N)[1:]),
                       "success_seeds":sorted([s for s in seeds if ma[s]])},
      "ref_5090_success":{"k":kr,"n":N,"rate":wilson(kr,N)[0]},
      "agreement_vs_5090":{"n_agree":len(agree),"n_flips":len(dis),"flip_seeds":dis,
                           "flip_detail":[{"seed":s,"arm":ma[s],"5090":mr[s]} for s in dis],
                           "agreement_rate":round(len(agree)/N,4) if N else 0},
      "fidelity_arm_joints_vs_5090":{"pooled_rel_mae":round(pooled,5),
          "per_seed":per,"distribution":{"min":round(float(np.min(vals)),5),
          "median":round(float(np.median(vals)),5),"mean":round(float(np.mean(vals)),5),
          "max":round(float(np.max(vals)),5)}},
      "latency":calllog_lat(calllog),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ref5090",required=True)
    ap.add_argument("--fp16_dir",default=None); ap.add_argument("--fp16_calllog",default=None)
    ap.add_argument("--armA_dir",required=True); ap.add_argument("--armA_calllog",required=True)
    ap.add_argument("--armB_dir",required=True); ap.add_argument("--armB_calllog",required=True)
    ap.add_argument("--armC_dir",default=None); ap.add_argument("--armC_calllog",default=None)
    ap.add_argument("--gate",required=True); ap.add_argument("--out",required=True)
    args=ap.parse_args()
    gate=json.load(open(args.gate))
    out={"experiment":"iq9 Hexagon v73 int8-vs-fp16 GR00T precision ladder, closed-loop "
                    "RoboCasa (same harness as closed_loop_m5, shared noise bank bit-matched, vs 5090 bf16)",
         "tag":"MEASURED",
         "quantization_recipe":{"tool":"qairt-quantizer 2.47","scheme":"W8A8 per-channel weights, "
             "asymmetric per-tensor activations (uint8, non-zero zero-point => negatives preserved, "
             "avoids the signed->unsigned clip trap), bias 8b","calibration":"45 REAL in-loop board "
             "frames (2 scenes) for body graphs / 80 (20 frames x 4 denoise steps) for head graphs, "
             "captured from the 5090 FT policy rollout via GrootPrep -> fp16 ONNX per-graph inputs",
             "io":"FLOAT_32 IO preserved (--preserve_io_datatype) so resident_orch float chaining "
                  "runs the int8 graphs unchanged; internal compute genuine W8A8"},
         "per_graph_int8_vs_fp16_gate":gate,
         "arms":{}}
    out["arms"]["armA_int8body_fp16head"]=arm_metrics(args.armA_dir,args.ref5090,args.armA_calllog)
    out["arms"]["armB_int8_everything"]=arm_metrics(args.armB_dir,args.ref5090,args.armB_calllog)
    if args.armC_dir:
        out["arms"]["armC_fp16body_int8head"]=arm_metrics(args.armC_dir,args.ref5090,args.armC_calllog)
    if args.fp16_dir:
        out["arms"]["fp16_baseline"]=arm_metrics(args.fp16_dir,args.ref5090,args.fp16_calllog)
    json.dump(out,open(args.out,"w"),indent=1)
    print(json.dumps({k:{"success":v["success"],"flips":v["agreement_vs_5090"]["n_flips"],
        "fid_median":v["fidelity_arm_joints_vs_5090"]["distribution"]["median"],
        "wall_ms":(v["latency"] or {}).get("board_wall_ms_median")}
        for k,v in out["arms"].items()},indent=1))
    print("WROTE",args.out)

if __name__=="__main__": main()
