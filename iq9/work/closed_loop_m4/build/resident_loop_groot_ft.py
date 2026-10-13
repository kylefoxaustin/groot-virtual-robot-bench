#!/usr/bin/env python3
"""M4 pre-loop GATE (board-side): FT bins in ctx_ft, FT glue in glue_ft, inputs in io_ft.
Runs one frame via per-graph EXEC (resident_orch), compares each rebuilt graph's board
fp16 output to the host FT-fp16-ONNX ref (io_ft/ref_*.npy) -> per-graph rel_mae/cos, and
the e2e action vs io_ft/host_e2e_ft.npy. --null-image zeroes pixel_values for the null arm.
Ports resident_loop_groot_fix.py; only paths + per-graph gate added.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
import os, sys, time, subprocess, json, glob, argparse
import numpy as np
sys.path.insert(0, _B+"/optionB/src")
from host_glue_groot import GrootGlue

BROOT = _B+"/optionB"
CTX = f"{BROOT}/ctx_ft"; IO = f"{BROOT}/io_ft"; GLUE = f"{BROOT}/glue_ft"
QNN = os.environ.get("QNN_ROOT", _BQ+"")
os.environ["GROOT_GLUE"] = GLUE

CTX_BINS = {
    f"{CTX}/groot_vit_fp16.bin": "groot_vit",
    f"{CTX}/groot_llm_fp16.bin": "groot_llm",
    f"{CTX}/vlln_vl_self_attention_fp16.bin": "vlln_vl_self_attention_47",
    f"{CTX}/state_encoder_fp16.bin": "state_encoder_47",
    f"{CTX}/action_encoder_fp16.bin": "action_encoder_47",
    f"{CTX}/groot_dit_fp16.bin": "groot_dit",
    f"{CTX}/action_decoder_fp16.bin": "action_decoder_47",
}
GN = {"vit":"groot_vit","llm":"groot_llm","vlln":"vlln_vl_self_attention_47",
      "state_encoder":"state_encoder_47","action_encoder":"action_encoder_47",
      "DiT":"groot_dit","action_decoder":"action_decoder_47"}
PROTO = ("LOADED","READY","OK","PONG","BYE","ERR")
IMAGE_TOKEN_INDEX = 151669

class Resident:
    def __init__(self, outdir, log_level="error"):
        self.outdir = outdir; self._dirs=set(); os.makedirs(outdir, exist_ok=True)
        env=dict(os.environ); env["LD_LIBRARY_PATH"]=f"{QNN}/lib:"+env.get("LD_LIBRARY_PATH","")
        env["ADSP_LIBRARY_PATH"]=f"{QNN}/skel"
        args=[f"{BROOT}/resident/resident_orch","--backend",f"{QNN}/lib/libQnnHtp.so",
              "--system_library",f"{QNN}/lib/libQnnSystem.so","--log_level",log_level]
        for path in CTX_BINS: args += ["--ctx", path]
        self.errlog=open(f"{outdir}/orch_stderr.log","w")
        self.p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                stderr=self.errlog,env=env,text=True,bufsize=1)
        self.loaded=[]; self.load_us=None
        while True:
            line=self.p.stdout.readline()
            if not line: raise RuntimeError("orch died during load; see orch_stderr.log")
            line=line.strip()
            if line.startswith("LOADED"): self.loaded.append(line.split()[1])
            elif line.startswith("READY"):
                for t in line.split():
                    if t.startswith("load_us="): self.load_us=int(t.split("=")[1])
                break
            elif line.startswith("ERR"): raise RuntimeError(f"load error: {line}")
    def _readline_proto(self):
        while True:
            line=self.p.stdout.readline()
            if not line: raise RuntimeError("orch closed stdout")
            line=line.strip()
            if line.startswith(PROTO): return line
    def exec_graph(self,name,inputs,tag):
        il=f"{self.outdir}/il_{tag}.txt"
        with open(il,"w") as f: f.write(" ".join(f"{k}:={v}" for k,v in inputs.items())+"\n")
        odir=f"{self.outdir}/{tag}"
        if tag not in self._dirs: os.makedirs(f"{odir}/Result_0",exist_ok=True); self._dirs.add(tag)
        self.p.stdin.write(f"EXEC {name} {il} {odir}\n"); self.p.stdin.flush()
        resp=self._readline_proto()
        if not resp.startswith("OK"): raise RuntimeError(f"exec {name} failed: {resp}")
        outs={os.path.basename(f)[:-4]:np.fromfile(f,np.float32) for f in glob.glob(f"{odir}/Result_0/*.raw")}
        return outs
    def close(self):
        try: self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(timeout=30)
        except Exception: self.p.terminate()
        self.errlog.close()

def w(path,arr,dt=np.float32): arr.astype(dt).tofile(path); return path

def run_frame(R,g,pixel_path,cap):
    o=R.exec_graph(GN["vit"],{"pixel_values":pixel_path,"position_ids":f"{IO}/position_ids.raw"},"vit")
    vit=list(o.values())[0].reshape(1,256,1152); cap["vit"]=vit
    proj=g.mlp1_project(vit)
    input_ids=np.fromfile(f"{IO}/input_ids.raw",np.int64).reshape(1,-1)
    ie=g.embed_and_scatter(input_ids,proj,IMAGE_TOKEN_INDEX); N=ie.shape[1]
    w(f"{IO}/_ie.raw",ie)
    o=R.exec_graph(GN["llm"],{"inputs_embeds":f"{IO}/_ie.raw","attention_mask":f"{IO}/attention_mask.raw"},"llm")
    backbone=list(o.values())[0].reshape(1,N,2048); w(f"{IO}/_bb.raw",backbone)
    o=R.exec_graph(GN["vlln"],{"backbone_features":f"{IO}/_bb.raw"},"vlln")
    vl=list(o.values())[0].reshape(1,-1,2048); cap["vlln"]=vl; w(f"{IO}/_vl.raw",vl)
    o=R.exec_graph(GN["state_encoder"],{"state":f"{IO}/state.raw"},"state")
    sf=list(o.values())[0].reshape(1,1,1536); cap["state"]=sf
    actions=np.fromfile(f"{IO}/init_actions.raw",np.float32).reshape(1,16,32)
    for s in range(g.num_steps):
        td=g.t_discretized(s); w(f"{IO}/_act.raw",actions); w(f"{IO}/_ts.raw",np.array([td],np.float32))
        o=R.exec_graph(GN["action_encoder"],{"actions":f"{IO}/_act.raw","time_proj":f"{IO}/time_proj_t{td}.raw"},"aenc")
        araw=list(o.values())[0].reshape(1,16,1536)
        af=g.add_action_pos_embed(araw); sa=g.assemble_sa_embs(sf,af); w(f"{IO}/_sa.raw",sa)
        o=R.exec_graph(GN["DiT"],{"sa_embs":f"{IO}/_sa.raw","vl_embs":f"{IO}/_vl.raw","timesteps_tensor":f"{IO}/_ts.raw"},"dit")
        mo=list(o.values())[0].reshape(1,49,1024); w(f"{IO}/_mo.raw",mo)
        o=R.exec_graph(GN["action_decoder"],{"model_output":f"{IO}/_mo.raw"},"adec")
        pred=list(o.values())[0].reshape(1,49,32)
        if s==g.num_steps-1: cap["aenc"]=araw; cap["dit"]=mo; cap["adec"]=pred
        actions=g.euler(actions,pred)
    return actions

def gate(a,b):
    a=a.astype(np.float64).ravel(); b=b.astype(np.float64).ravel()
    rel=float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
    cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    return rel,cos

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default=f"{BROOT}/out/m4_gate")
    ap.add_argument("--null-image",action="store_true")
    args=ap.parse_args()
    g=GrootGlue(GLUE)
    pixel_path=f"{IO}/pixel_values.raw"
    if args.null_image:
        pv=np.zeros_like(np.fromfile(f"{IO}/pixel_values.raw",np.float32)); pixel_path=f"{IO}/_pixnull.raw"; w(pixel_path,pv)
    R=Resident(args.outdir)
    print(f"[m4-gate] loaded {len(R.loaded)} graphs: {R.loaded} ctx_load={R.load_us/1e3:.1f}ms",flush=True)
    cap={}
    try: act=run_frame(R,g,pixel_path,cap)
    finally: R.close()
    out={"null_image":args.null_image,"n_graphs":len(R.loaded),"per_graph":{}}
    for k in ["vit","vlln","state","aenc","dit","adec"]:
        ref_p=f"{IO}/ref_{k}.npy"
        if os.path.exists(ref_p) and k in cap:
            r,c=gate(cap[k], np.load(ref_p)); out["per_graph"][k]={"rel_mae":round(r,5),"cos":round(c,6)}
    gp=f"{IO}/host_e2e_ft.npy"
    if os.path.exists(gp):
        gold=np.load(gp).reshape(1,16,32)
        r,c=gate(act,gold); r26,_=gate(act[...,:26],gold[...,:26])
        out["e2e_vs_host"]={"rel_mae_32":round(r,5),"rel_mae_26real":round(r26,5),"cos":round(c,6),
                            "fp16_floor_5e-3":r<5e-3,"bf16_gate_0.04":r<0.04}
    np.save(f"{args.outdir}/action_ft{'_null' if args.null_image else ''}.npy",act)
    print(json.dumps(out,indent=1),flush=True); print("DONE_M4_GATE",flush=True)

if __name__=="__main__": main()
