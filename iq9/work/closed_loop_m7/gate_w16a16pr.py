#!/usr/bin/env python3
"""W16A16 + per-row weights gate: for the 3 body graphs compare the W16A16-per-row board
output vs fp16 (cosine, rel_mae; PASS = cos > 0.95). Compare against the W8A16 gate to see
whether 8-bit weights or the quantization path is the limiting factor."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, subprocess, json, numpy as np
BOARD=_gvrb_os.environ.get("IQ9_HOST", "iq9"); QNN=_BQ+""; BROOT=_B+"/optionB"
GATE=f"{BROOT}/gate_m7"; CTX_FP=f"{BROOT}/ctx_ft"; CTX_I8=f"{BROOT}/ctx_int8"
HOST_CTX=_W+"/closed_loop_m7/int8_build/ctx"
OUTJSON=_W+"/closed_loop_m7/int8_build/w16a16pr_gate.json"
GRAPHS={
 "groot_vit":{"in":[("pixel_values","vit__pixel_values.raw"),("position_ids","vit__position_ids.raw")],"out":"vit_embeds"},
 "groot_llm":{"in":[("inputs_embeds","llm__inputs_embeds.raw"),("attention_mask","llm__attention_mask.raw")],"out":"embeddings"},
}
def sh(c,**k): return subprocess.run(c,shell=True,text=True,capture_output=True,**k)
def bssh(c,**k): return sh(f"ssh {BOARD} '{c}'",**k)
def run_graph(g,binp,tag):
    d=f"{GATE}/{tag}_{g}"
    if bssh(f"ls {d}/out/Result_0/ 2>/dev/null | head -1").stdout.strip(): return d
    inl=" ".join(f"{n}:={GATE}/in/{f}" for n,f in GRAPHS[g]["in"])
    bssh(f"rm -rf {d}; mkdir -p {d}; echo '{inl}' > {d}/list.txt")
    bssh(f"cd {d} && LD_LIBRARY_PATH={QNN}/lib ADSP_LIBRARY_PATH={QNN}/skel {QNN}/bin/qnn-net-run "
         f"--backend {QNN}/lib/libQnnHtp.so --retrieve_context {binp} --input_list {d}/list.txt --output_dir {d}/out")
    return d
def pull(d,outn):
    lf=f"/tmp/w16_{os.path.basename(d)}.raw"; files=bssh(f"ls {d}/out/Result_0/ 2>/dev/null").stdout.split()
    if not files: return None
    fn=next((x for x in files if x.startswith(outn)),files[0]); sh(f"scp -q {BOARD}:{d}/out/Result_0/{fn} {lf}")
    return np.fromfile(lf,np.float32) if os.path.exists(lf) else None
cos=lambda a,b: float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
rmae=lambda a,b: float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def main():
    for g in GRAPHS:
        b=f"{HOST_CTX}/{g}_w16a16pr.bin"
        if os.path.exists(b) and bssh(f"stat -c%s {CTX_I8}/{g}_w16a16pr.bin 2>/dev/null").stdout.strip()!=str(os.path.getsize(b)):
            print(f"push {g}_w16a16pr.bin",flush=True); sh(f"scp -q {b} {BOARD}:{CTX_I8}/{g}_w16a16pr.bin")
    res={}
    for g in GRAPHS:
        fpn=g[:-3] if g.endswith("_47") else g
        o_fp=pull(run_graph(g,f"{CTX_FP}/{fpn}_fp16.bin","fp"),GRAPHS[g]["out"])
        o_16=pull(run_graph(g,f"{CTX_I8}/{g}_w16a16pr.bin","w16a16pr"),GRAPHS[g]["out"])
        if o_fp is None or o_16 is None: res[g]={"error":f"fp={o_fp is not None} w16={o_16 is not None}"};
        else:
            n=min(len(o_fp),len(o_16)); a,b=o_16[:n],o_fp[:n]
            res[g]={"cosine":round(cos(a,b),6),"rel_mae":round(rmae(a,b),5),"PASS_cos0.95":bool(cos(a,b)>0.95)}
            print(f"[w16a16pr] {g:28s} cos={cos(a,b):.5f} rel_mae={rmae(a,b):.4f}",flush=True)
        json.dump(res,open(OUTJSON,"w"),indent=1)
    print("WROTE",OUTJSON)
if __name__=="__main__": main()
