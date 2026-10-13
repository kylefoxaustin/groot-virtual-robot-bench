#!/usr/bin/env python3
"""closed_loop_m8 SmoothQuant per-graph gate. Push each groot_llm variant bin to the board, run
qnn-net-run on the SAME held-out real gate input closed_loop_m7 used, compare embeddings vs the fp16
board reference. Baseline to beat: the unsmoothed control variant (ctrl_a16, W8A16 per-row) in the same gate."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, subprocess, json, numpy as np
BOARD=_gvrb_os.environ.get("IQ9_HOST", "iq9"); QNN=_BQ+""; BROOT=_B+"/optionB"
GATE=f"{BROOT}/gate_m7"; CTX_FP=f"{BROOT}/ctx_ft"; CTX_M8=f"{BROOT}/ctx_m8"
HOST_CTX=_W+"/closed_loop_m8/ctx"
OUTJSON=_W+"/closed_loop_m8/m8_pergraph_gate.json"
IN=[("inputs_embeds","llm__inputs_embeds.raw"),("attention_mask","llm__attention_mask.raw")]
OUTNAME="embeddings"
VARIANTS=["ctrl_a16","ctrl_a8","a50_a16","a50_a8","a85_a16","a85_a8"]

def sh(c,**k): return subprocess.run(c,shell=True,text=True,capture_output=True,**k)
def bssh(c,**k): return sh(f"ssh {BOARD} '{c}'",**k)
def run_graph(binp,tag):
    d=f"{GATE}/m8_{tag}"
    if bssh(f"ls {d}/out/Result_0/ 2>/dev/null | head -1").stdout.strip(): return d,"REUSED"
    inl=" ".join(f"{n}:={GATE}/in/{f}" for n,f in IN)
    bssh(f"rm -rf {d}; mkdir -p {d}; echo '{inl}' > {d}/list.txt")
    r=bssh(f"cd {d} && LD_LIBRARY_PATH={QNN}/lib ADSP_LIBRARY_PATH={QNN}/skel {QNN}/bin/qnn-net-run "
           f"--backend {QNN}/lib/libQnnHtp.so --retrieve_context {binp} --input_list {d}/list.txt --output_dir {d}/out")
    return d, (r.stdout+r.stderr)[-400:]
def pull(d):
    lf=f"/tmp/m8_{os.path.basename(d)}.raw"; files=bssh(f"ls {d}/out/Result_0/ 2>/dev/null").stdout.split()
    if not files: return None
    fn=next((x for x in files if x.startswith(OUTNAME)),files[0]); sh(f"scp -q {BOARD}:{d}/out/Result_0/{fn} {lf}")
    return np.fromfile(lf,np.float32) if os.path.exists(lf) else None
cos=lambda a,b: float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
rmae=lambda a,b: float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def stats(o): return {"mean_abs":round(float(np.mean(np.abs(o))),6),"frac_zero":round(float((o==0).mean()),4)}

def main():
    bssh(f"mkdir -p {CTX_M8}")
    # fp16 ref
    d_fp,log=run_graph(f"{CTX_FP}/groot_llm_fp16.bin","fp"); o_fp=pull(d_fp)
    assert o_fp is not None, f"fp16 ref missing: {log}"
    print(f"fp16 ref: n={len(o_fp)} mean_abs={np.mean(np.abs(o_fp)):.4f}")
    res={"fp16_ref":{**stats(o_fp),"n":int(len(o_fp))},"variants":{}}
    for v in VARIANTS:
        b=f"{HOST_CTX}/groot_llm_{v}.bin"
        if not os.path.exists(b): print(f"[skip] {v} bin absent on host"); continue
        # push if size differs
        rem=bssh(f"stat -c%s {CTX_M8}/groot_llm_{v}.bin 2>/dev/null").stdout.strip()
        if rem!=str(os.path.getsize(b)):
            print(f"push {v} ({os.path.getsize(b)} B)...",flush=True); sh(f"scp -q {b} {BOARD}:{CTX_M8}/groot_llm_{v}.bin")
        d,log=run_graph(f"{CTX_M8}/groot_llm_{v}.bin",v); o=pull(d)
        if o is None: res["variants"][v]={"status":"missing_out","log":log}; print(f"[gate] {v}: MISSING\n{log}"); continue
        n=min(len(o),len(o_fp))
        c=cos(o[:n],o_fp[:n])
        res["variants"][v]={"cosine":round(c,6),"rel_mae":round(rmae(o[:n],o_fp[:n]),5),
                            "out":stats(o),"PASS_cos0.95":bool(c>0.95)}
        print(f"[gate] {v:9s} cos={c:.5f} rel_mae={rmae(o[:n],o_fp[:n]):.4f} out_meanabs={np.mean(np.abs(o)):.4f}",flush=True)
        json.dump(res,open(OUTJSON,"w"),indent=1)
    json.dump(res,open(OUTJSON,"w"),indent=1); print("WROTE",OUTJSON)

if __name__=="__main__": main()
