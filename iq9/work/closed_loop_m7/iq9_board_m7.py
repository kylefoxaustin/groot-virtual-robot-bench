#!/usr/bin/env python3
"""M7 board transport = iq9_board_m4 with a CONFIGURABLE ctx-bin set (env M7_CTX_LIST,
comma-separated absolute board paths to the 7 ctx bins, any fp16/int8 mix). Everything
else (resident GFRAME protocol, glue_ft, io_ft, clean-QUIT safety) is identical to M4.

Int8 bins are built with FLOAT IO preserved, so the resident_orch's float graph->graph
memcpy chaining consumes them UNCHANGED (internal compute is genuine W8A8)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
_BP = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_PY_ROOT", "/opt/iq9-python"))
import os, sys, time, subprocess, tempfile
import numpy as np

BOARD=os.environ.get("IQ9_HOST","iq9"); QNN=_BQ+""; BROOT=_B+"/optionB"
BLAS=_BP+"/python/lib/python3.12/site-packages/torch/lib/libopenblas.so.0"
GLUE=f"{BROOT}/glue_ft"; SRC_IO=f"{BROOT}/io_ft"; IO_M2=f"{BROOT}/io_m7"
# default = M4 fp16 baseline
_DEF=[f"{BROOT}/ctx_ft/{b}" for b in
      ["groot_vit_fp16.bin","groot_llm_fp16.bin","vlln_vl_self_attention_fp16.bin",
       "state_encoder_fp16.bin","action_encoder_fp16.bin","groot_dit_fp16.bin","action_decoder_fp16.bin"]]
CTX_BINS=os.environ.get("M7_CTX_LIST","").split(",") if os.environ.get("M7_CTX_LIST") else _DEF
PROTO=("LOADED","READY","OK","OKGFRAME","PONG","BYE","ERR")

class IQ9Board:
    def __init__(self, workdir, blas_threads=6, log_level="error"):
        self.workdir=workdir; os.makedirs(workdir,exist_ok=True)
        self.ctl=f"/tmp/.m7ssh_{os.getpid()}.sock"
        self._ssh_base=["ssh","-o","ControlMaster=auto","-o",f"ControlPath={self.ctl}",
                        "-o","ControlPersist=300","-o","ServerAliveInterval=30"]
        subprocess.run(self._ssh_base+["-fN",BOARD],check=True)
        self._setup_board_io(); self._launch_resident(blas_threads,log_level)
    def _ssh(self,cmd,**kw): return subprocess.run(self._ssh_base+[BOARD,cmd],**kw)
    def _scp(self,lf,rd):
        subprocess.run(["scp","-o",f"ControlPath={self.ctl}","-q"]+lf+[f"{BOARD}:{rd}/"],check=True)
    def _setup_board_io(self):
        self._ssh(f"mkdir -p {IO_M2} && cp -f {SRC_IO}/time_proj_t*.raw {IO_M2}/",check=True)
    def stage_constants(self, ids, am, pid):
        d=tempfile.mkdtemp(dir=self.workdir)
        np.ascontiguousarray(ids,np.int64).tofile(f"{d}/input_ids.raw")
        np.ascontiguousarray(am,np.float32).tofile(f"{d}/attention_mask.raw")
        np.ascontiguousarray(pid,np.float32).tofile(f"{d}/position_ids.raw")
        self._scp([f"{d}/input_ids.raw",f"{d}/attention_mask.raw",f"{d}/position_ids.raw"],IO_M2)
    def _launch_resident(self, blas_threads, log_level):
        args=[f"{BROOT}/resident/resident_orch","--backend",f"{QNN}/lib/libQnnHtp.so",
              "--system_library",f"{QNN}/lib/libQnnSystem.so","--log_level",log_level,
              "--blas",BLAS,"--glue",GLUE,"--blas_threads",str(blas_threads)]
        for b in CTX_BINS: args+=["--ctx",b]
        remote=(f"cd {BROOT}/resident && LD_LIBRARY_PATH={QNN}/lib ADSP_LIBRARY_PATH={QNN}/skel "+" ".join(args))
        self.errlog=open(os.path.join(self.workdir,"orch_stderr.log"),"w")
        self.p=subprocess.Popen(self._ssh_base+[BOARD,remote],stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=self.errlog,text=True,bufsize=1)
        self.loaded=[]; self.load_us=None; t0=time.time()
        while True:
            line=self.p.stdout.readline()
            if not line: raise RuntimeError("resident_orch died during load; see orch_stderr.log")
            line=line.strip()
            if line.startswith("LOADED"): self.loaded.append(line.split()[1])
            elif line.startswith("READY"):
                for t in line.split():
                    if t.startswith("load_us="): self.load_us=int(t.split("=")[1])
                break
            elif line.startswith("ERR"): raise RuntimeError(f"load error: {line}")
        self.load_wall=time.time()-t0
    def _readline_proto(self):
        while True:
            line=self.p.stdout.readline()
            if not line: raise RuntimeError("resident_orch closed stdout; see orch_stderr.log")
            line=line.strip()
            if line.startswith(PROTO): return line
    def gframe(self, pixel, state, init_actions):
        d=tempfile.mkdtemp(dir=self.workdir)
        np.ascontiguousarray(pixel,np.float32).tofile(f"{d}/pixel_values.raw")
        np.ascontiguousarray(state,np.float32).tofile(f"{d}/state.raw")
        np.ascontiguousarray(init_actions,np.float32).tofile(f"{d}/init_actions.raw")
        t0=time.time(); self._scp([f"{d}/pixel_values.raw",f"{d}/state.raw",f"{d}/init_actions.raw"],IO_M2)
        push_s=time.time()-t0; t1=time.time()
        self.p.stdin.write(f"GFRAME {IO_M2} {IO_M2}/action.raw\n"); self.p.stdin.flush()
        resp=self._readline_proto()
        if not resp.startswith("OKGFRAME"): raise RuntimeError(f"GFRAME failed: {resp}")
        gframe_s=time.time()-t1; tm={}
        for tok in resp.split()[1:]:
            k,v=tok.split("="); tm[k]=int(v)
        lf=os.path.join(d,"action.raw")
        subprocess.run(["scp","-o",f"ControlPath={self.ctl}","-q",f"{BOARD}:{IO_M2}/action.raw",lf],check=True)
        action=np.fromfile(lf,np.float32).reshape(1,16,32); wall_s=time.time()-t0
        tm["_push_s"]=push_s; tm["_gframe_s"]=gframe_s; tm["_wall_s"]=wall_s
        return action,tm
    def health(self):
        r=self._ssh("for f in /sys/class/remoteproc/remoteproc*/state; do cat $f; done; "
                    "ps -ef | grep -E 'resident_orch|qnn-net-run' | grep -v grep | wc -l",
                    capture_output=True,text=True); return r.stdout.strip()
    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(timeout=30)
        except Exception:
            try: self.p.terminate(); self.p.wait(timeout=10)
            except Exception: pass
        try: self.errlog.close()
        except Exception: pass
        subprocess.run(self._ssh_base+["-O","exit",BOARD],capture_output=True)
