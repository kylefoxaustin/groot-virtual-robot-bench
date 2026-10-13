#!/usr/bin/env python3
"""Body-precision discriminating gate. For the 3 body graphs (vit/llm/vlln) run the
fp16 board bin AND each int8 variant (min-max W8A8, percentile W8A8, W8A16) through
qnn-net-run on the SAME held-out real input, compare each variant's output vs fp16.
Separates 8-bit-activation precision loss (W8A16 recovers where W8A8 does not) from a
build problem (W8A16 also fails)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, subprocess, json, numpy as np, sys

BOARD=_gvrb_os.environ.get("IQ9_HOST", "iq9"); QNN=_BQ+""; BROOT=_B+"/optionB"
GATE=f"{BROOT}/gate_m7"; CTX_FP=f"{BROOT}/ctx_ft"; CTX_I8=f"{BROOT}/ctx_int8"
HOST_IN=_W+"/closed_loop_m7/int8_build/gate_in"
HOST_CTX=_W+"/closed_loop_m7/int8_build/ctx"
OUTJSON=_W+"/closed_loop_m7/int8_build/body_variants_gate.json"

GRAPHS={
 "groot_vit":{"in":[("pixel_values","vit__pixel_values.raw"),("position_ids","vit__position_ids.raw")],"out":"vit_embeds"},
 "groot_llm":{"in":[("inputs_embeds","llm__inputs_embeds.raw"),("attention_mask","llm__attention_mask.raw")],"out":"embeddings"},
 "vlln_vl_self_attention_47":{"in":[("backbone_features","vlln__backbone_features.raw")],"out":"output"},
 "state_encoder_47":{"in":[("state","state__state.raw")],"out":"output"},
 "action_encoder_47":{"in":[("actions","aenc__actions.raw"),("time_proj","aenc__time_proj.raw")],"out":"output"},
 "groot_dit":{"in":[("sa_embs","dit__sa_embs.raw"),("vl_embs","dit__vl_embs.raw"),("timesteps_tensor","dit__timesteps_tensor.raw")],"out":"output"},
 "action_decoder_47":{"in":[("model_output","adec__model_output.raw")],"out":"output"},
}
# variant -> board bin basename suffix (relative to CTX_I8), fp16 handled separately
VARIANTS={"int8_minmax":"_int8","int8_pctl":"_int8p","w8a16":"_a16"}

def sh(c,**k): return subprocess.run(c,shell=True,text=True,capture_output=True,**k)
def bssh(c,**k): return sh(f"ssh {BOARD} '{c}'",**k)

def run_graph(gname, binpath, tag):
    d=f"{GATE}/{tag}_{gname}"
    # RESUME: reuse an existing board result (avoids reloading big bins -> less host PSI pressure)
    if bssh(f"ls {d}/out/Result_0/ 2>/dev/null | head -1").stdout.strip():
        return d, "REUSED_EXISTING"
    inl=" ".join(f"{n}:={GATE}/in/{f}" for n,f in GRAPHS[gname]["in"])
    bssh(f"rm -rf {d}; mkdir -p {d}; echo '{inl}' > {d}/list.txt")
    cmd=(f"cd {d} && LD_LIBRARY_PATH={QNN}/lib ADSP_LIBRARY_PATH={QNN}/skel "
         f"{QNN}/bin/qnn-net-run --backend {QNN}/lib/libQnnHtp.so --retrieve_context {binpath} "
         f"--input_list {d}/list.txt --output_dir {d}/out")
    r=bssh(cmd)
    return d, r.stdout+r.stderr

def pull_out(d, outname):
    lf=f"/tmp/m7bv_{os.path.basename(d)}.raw"
    r=bssh(f"ls {d}/out/Result_0/ 2>/dev/null"); files=r.stdout.split()
    if not files: return None
    fn=next((x for x in files if x.startswith(outname)), files[0])
    sh(f"scp -q {BOARD}:{d}/out/Result_0/{fn} {lf}")
    return np.fromfile(lf, np.float32) if os.path.exists(lf) else None

def relmae(a,b): return float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def stats(o):
    return {"mean_abs":round(float(np.mean(np.abs(o))),6),"std":round(float(o.std()),6),
            "frac_zero":round(float((o==0).mean()),4),"frac_neg":round(float((o<0).mean()),4)}

def main():
    bssh(f"mkdir -p {GATE}/in {CTX_I8}")
    sh(f"scp -q {HOST_IN}/*.raw {BOARD}:{GATE}/in/")
    # push any a16 bins present on host
    for g in GRAPHS:
        for v,suf in VARIANTS.items():
            b=f"{HOST_CTX}/{g}{suf}.bin"
            if os.path.exists(b):
                exists=bssh(f"test -f {CTX_I8}/{g}{suf}.bin && stat -c%s {CTX_I8}/{g}{suf}.bin").stdout.strip()
                if exists != str(os.path.getsize(b)):
                    print(f"push {g}{suf}.bin ...",flush=True); sh(f"scp -q {b} {BOARD}:{CTX_I8}/{g}{suf}.bin")
    res={}
    for g in GRAPHS:
        fpname=g[:-3] if g.endswith("_47") else g
        fp_bin=f"{CTX_FP}/{fpname}_fp16.bin"
        d_fp,log_fp=run_graph(g,fp_bin,"fp"); o_fp=pull_out(d_fp,GRAPHS[g]["out"])
        if o_fp is None:
            print(f"[gate] {g}: FP16 MISSING OUTPUT\n{log_fp[-500:]}"); res[g]={"error":"fp16_missing"}; continue
        entry={"fp16_out":{**stats(o_fp),"n":int(len(o_fp))},"variants":{}}
        for v,suf in VARIANTS.items():
            i8_bin=f"{CTX_I8}/{g}{suf}.bin"
            if bssh(f"test -f {i8_bin} && echo Y").stdout.strip()!="Y":
                entry["variants"][v]={"status":"bin_absent"}; continue
            d_i8,log_i8=run_graph(g,i8_bin,f"{v}"); o_i8=pull_out(d_i8,GRAPHS[g]["out"])
            if o_i8 is None:
                entry["variants"][v]={"status":"missing_output","log":log_i8[-300:]}; continue
            n=min(len(o_fp),len(o_i8)); a,b=o_i8[:n],o_fp[:n]
            entry["variants"][v]={"rel_mae":round(relmae(a,b),5),"cosine":round(cos(a,b),6),
                                  "max_abs_diff":round(float(np.max(np.abs(a-b))),5),
                                  "int8_out":stats(a),
                                  "PASS_cos0.95":bool(cos(a,b)>0.95)}
            print(f"[gate] {g:28s} {v:12s} cos={cos(a,b):.5f} rel_mae={relmae(a,b):.4f}",flush=True)
        res[g]=entry
        json.dump(res,open(OUTJSON,"w"),indent=1)  # incremental: survive an OOM kill
    json.dump(res,open(OUTJSON,"w"),indent=1); print("WROTE",OUTJSON)

if __name__=="__main__": main()
