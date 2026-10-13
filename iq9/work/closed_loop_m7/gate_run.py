#!/usr/bin/env python3
"""M7 per-graph accuracy gate ON THE BOARD: run each graph's fp16 ctx bin AND its int8
ctx bin through qnn-net-run on the SAME held-out real input, compare outputs.

Proves (a) int8 engaged (output differs from fp16 in the quant way, not identical, not
all-zero), (b) localizes precision loss per stage (esp. the DiT action head). rel_mae is
int8-vs-fp16 (fp16 = board's own reference). Also reports the fraction of negative
outputs, which exposes a signed->unsigned clip (negatives clipped to zero) if one occurs."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, subprocess, json, numpy as np

BOARD=_gvrb_os.environ.get("IQ9_HOST", "iq9"); QNN=_BQ+""; BROOT=_B+"/optionB"
GATE=f"{BROOT}/gate_m7"; CTX_FP=f"{BROOT}/ctx_ft"; CTX_I8=f"{BROOT}/ctx_int8"
HOST_IN=_W+"/closed_loop_m7/int8_build/gate_in"
HOST_CTX=_W+"/closed_loop_m7/int8_build/ctx"
OUTJSON=_W+"/closed_loop_m7/int8_build/pergraph_gate.json"

GRAPHS={
 "groot_vit":{"in":[("pixel_values","vit__pixel_values.raw"),("position_ids","vit__position_ids.raw")],"out":"vit_embeds"},
 "groot_llm":{"in":[("inputs_embeds","llm__inputs_embeds.raw"),("attention_mask","llm__attention_mask.raw")],"out":"embeddings"},
 "vlln_vl_self_attention_47":{"in":[("backbone_features","vlln__backbone_features.raw")],"out":"output"},
 "state_encoder_47":{"in":[("state","state__state.raw")],"out":"output"},
 "action_encoder_47":{"in":[("actions","aenc__actions.raw"),("time_proj","aenc__time_proj.raw")],"out":"output"},
 "groot_dit":{"in":[("sa_embs","dit__sa_embs.raw"),("vl_embs","dit__vl_embs.raw"),("timesteps_tensor","dit__timesteps_tensor.raw")],"out":"output"},
 "action_decoder_47":{"in":[("model_output","adec__model_output.raw")],"out":"output"},
}
def sh(c,**k): return subprocess.run(c,shell=True,text=True,capture_output=True,**k)
def bssh(c,**k): return sh(f"ssh {BOARD} '{c}'",**k)

def run_graph(gname, binpath, tag):
    d=f"{GATE}/{tag}_{gname}"
    inl=" ".join(f"{n}:={GATE}/in/{f}" for n,f in GRAPHS[gname]["in"])
    bssh(f"rm -rf {d}; mkdir -p {d}; echo '{inl}' > {d}/list.txt")
    cmd=(f"cd {d} && LD_LIBRARY_PATH={QNN}/lib ADSP_LIBRARY_PATH={QNN}/skel "
         f"{QNN}/bin/qnn-net-run --backend {QNN}/lib/libQnnHtp.so --retrieve_context {binpath} "
         f"--input_list {d}/list.txt --output_dir {d}/out")
    r=bssh(cmd)
    ok="error" not in (r.stdout+r.stderr).lower() or "Result_0" in bssh(f"ls {d}/out 2>/dev/null").stdout
    if not ok:
        raise RuntimeError(f"qnn-net-run failed for {gname} ({tag}, rc={r.returncode}):\n{(r.stdout+r.stderr)[-800:]}")
    return d, r.stdout+r.stderr

def pull_out(d, outname):
    lf=f"/tmp/m7_{os.path.basename(d)}.raw"
    # net-run writes out/Result_0/<outname>.raw
    r=bssh(f"ls {d}/out/Result_0/ 2>/dev/null")
    files=r.stdout.split()
    if not files: return None
    fn=next((x for x in files if x.startswith(outname)), files[0])
    sh(f"scp -q {BOARD}:{d}/out/Result_0/{fn} {lf}")
    if not os.path.exists(lf): return None
    return np.fromfile(lf, np.float32)

def relmae(a,b): return float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def main():
    # push inputs + int8 bins
    bssh(f"mkdir -p {GATE}/in {CTX_I8}")
    sh(f"scp -q {HOST_IN}/*.raw {BOARD}:{GATE}/in/")
    print("pushing int8 bins...", flush=True)
    for g in GRAPHS:
        b=f"{HOST_CTX}/{g}_int8.bin"
        if os.path.exists(b): sh(f"scp -q {b} {BOARD}:{CTX_I8}/{g}_int8.bin")
    res={}
    for g in GRAPHS:
        # fp16 bins on board are named WITHOUT the _47 suffix (deploy.sh renamed them)
        fpname=g[:-3] if g.endswith("_47") else g
        fp_bin=f"{CTX_FP}/{fpname}_fp16.bin"
        i8_bin=f"{CTX_I8}/{g}_int8.bin"
        exists=bssh(f"test -f {i8_bin} && echo Y").stdout.strip()
        if exists!="Y":
            print(f"[gate] {g}: int8 bin not on board yet, skip"); continue
        d_fp,log_fp=run_graph(g,fp_bin,"fp"); o_fp=pull_out(d_fp,GRAPHS[g]["out"])
        d_i8,log_i8=run_graph(g,i8_bin,"i8"); o_i8=pull_out(d_i8,GRAPHS[g]["out"])
        if o_fp is None or o_i8 is None:
            print(f"[gate] {g}: MISSING OUTPUT fp={o_fp is not None} i8={o_i8 is not None}")
            print("  fp log:",log_fp[-400:]); print("  i8 log:",log_i8[-400:]); res[g]={"error":"missing_output"}; continue
        n=min(len(o_fp),len(o_i8)); o_fp,o_i8=o_fp[:n],o_i8[:n]
        rm=relmae(o_i8,o_fp); cs=cos(o_i8,o_fp)
        res[g]={
          "int8_vs_fp16_rel_mae":round(rm,5),"cosine":round(cs,6),
          "max_abs_diff":round(float(np.max(np.abs(o_i8-o_fp))),5),
          "fp16_out":{"mean_abs":round(float(np.mean(np.abs(o_fp))),5),"std":round(float(o_fp.std()),5),
                      "frac_zero":round(float((o_fp==0).mean()),4),"n":int(n)},
          "int8_out":{"mean_abs":round(float(np.mean(np.abs(o_i8))),5),"std":round(float(o_i8.std()),5),
                      "frac_zero":round(float((o_i8==0).mean()),4),
                      "frac_neg":round(float((o_i8<0).mean()),4)},
          "engaged":bool(rm>1e-4 and (o_i8!=o_fp).any() and not np.all(o_i8==0)),
        }
        print(f"[gate] {g:30s} rel_mae={rm:.4f} cos={cs:.5f} i8_fracneg={res[g]['int8_out']['frac_neg']} engaged={res[g]['engaged']}",flush=True)
    json.dump(res,open(OUTJSON,"w"),indent=1); print("WROTE",OUTJSON)

if __name__=="__main__": main()
