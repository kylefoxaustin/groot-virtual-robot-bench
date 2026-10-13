#!/usr/bin/env python3
"""Assemble the closed_loop_m7 per-arm closed-loop metrics into the record JSON written by
finalize_m7.py, merging in place with the per-graph gate section already there. Uses the SAME metric defs
as finalize_m7 (success+Wilson, agreement vs 5090 + flips, arm-joint rel_mae dims0:14,
latency from server_calllog). Only includes arms whose rollout_summary.json exists."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, math, os, numpy as np
HERE=_W+"/closed_loop_m7"
REF=_W+"/closed_loop_m5/runs/r5090_20"
import sys
DELIV=sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"m7_record.json")   # record from finalize_m7.py (merged in place)

def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=(z/d)*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (round(p,4),round(max(0,c-h),4),round(min(1,c+h),4))
def summ(d): return json.load(open(os.path.join(d,"rollout_summary.json")))
def succ(s): return {e["seed"]:bool(e["success"]) for e in s["episodes"]}
def epidx(s): return {e["seed"]:e["episode"] for e in s["episodes"]}
def loadep(d,ep,seed): return np.load(os.path.join(d,f"episode_{ep:02d}_seed{seed}_actions.npy"))
def relmae(a,b,lo=0,hi=14):
    a=a.astype(np.float64)[...,lo:hi]; b=b.astype(np.float64)[...,lo:hi]
    return float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def latency(cl):
    if not os.path.exists(cl): return None
    c=json.load(open(cl)); md=lambda k:(round(float(np.median([x[k] for x in c if x.get(k) is not None])),1) if any(x.get(k) is not None for x in c) else None)
    o={"n_calls":len(c),"board_wall_ms_median":md("board_wall_ms")}
    for g in ["vit","llm","vlln","state","aenc","dit","adec_sum"]:
        v=[x[g] for x in c if x.get(g) is not None]
        if v: o[f"{g}_us_median"]=round(float(np.median(v)),1)
    return o
def arm(dirp, calllog):
    sa=summ(dirp); sr=summ(REF); ma=succ(sa); mr=succ(sr)
    seeds=sorted(set(ma)&set(mr)); N=len(seeds); ka=sum(ma[s] for s in seeds); kr=sum(mr[s] for s in seeds)
    dis=[s for s in seeds if ma[s]!=mr[s]]; eia=epidx(sa); eir=epidx(sr); per={}
    for s in seeds:
        a=loadep(dirp,eia[s],s); b=loadep(REF,eir[s],s); n=min(a.shape[0],b.shape[0]); per[str(s)]=round(relmae(a[:n],b[:n]),5)
    vals=list(per.values())
    p,lo,hi=wilson(ka,N)
    return {"N":N,"success":{"k":ka,"n":N,"rate":p,"wilson95":[lo,hi],"success_seeds":sorted([s for s in seeds if ma[s]])},
            "ref_5090_success":{"k":kr,"n":N,"rate":wilson(kr,N)[0]},
            "agreement_vs_5090":{"n_agree":N-len(dis),"n_flips":len(dis),"flip_seeds":dis,
                "flip_detail":[{"seed":s,"arm":ma[s],"5090":mr[s]} for s in dis],"agreement_rate":round((N-len(dis))/N,4) if N else 0},
            "fidelity_arm_joints_vs_5090":{"per_seed":per,"distribution":{"min":round(min(vals),5),"median":round(float(np.median(vals)),5),"mean":round(float(np.mean(vals)),5),"max":round(max(vals),5)} if vals else None},
            "latency":latency(calllog)}
ARMS={
 "fp16_baseline":("runs/fp16","runs/work/fp16/server_calllog.json"),
 "a16head_fp16body_W8A16head":("runs/a16head","runs/work/a16head/server_calllog.json"),
 "armC_fp16body_W8A8head":("runs/armC","runs/work/armC/server_calllog.json"),
 "prhead_fp16body_perROW_W8A16head":("runs/prhead","runs/work/prhead/server_calllog.json"),
}
d=json.load(open(DELIV)) if os.path.exists(DELIV) else {}; d["arms"]={}
for name,(dp,cl) in ARMS.items():
    dp=os.path.join(HERE,dp); cl=os.path.join(HERE,cl)
    if os.path.exists(os.path.join(dp,"rollout_summary.json")):
        try: d["arms"][name]=arm(dp,cl); print(f"{name}: success {d['arms'][name]['success']['k']}/{d['arms'][name]['success']['n']}  flips {d['arms'][name]['agreement_vs_5090']['n_flips']}  wall_ms {(d['arms'][name]['latency'] or {}).get('board_wall_ms_median')}")
        except Exception as e: print(f"{name}: ERR {e}")
    else: print(f"{name}: (no rollout yet)")
json.dump(d,open(DELIV,"w"),indent=1); print("WROTE",DELIV)
