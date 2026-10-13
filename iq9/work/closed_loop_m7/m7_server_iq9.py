#!/usr/bin/env python3
"""M7 iq9-in-the-loop server (int8/fp16 mix via M7_CTX_LIST). Identical semantics to
m4_server_iq9 (GrootPrep -> Hexagon v73 GFRAME loop -> unapply, injected fixed noise
bit-matched to the 5090). Only the board ctx-bin set differs (fp16 baseline / Arm A
int8-body+fp16-head / Arm B all-int8)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0,_W+"/closed_loop_m7")   # iq9_board_m7
sys.path.insert(0,_W+"/closed_loop_m2")   # iq9_prep
from iq9_prep import GrootPrep
from iq9_board_m7 import IQ9Board

class IQ9Shim:
    def __init__(self, model_path, work, noise_bank, data_config="fourier_gr1_arms_waist"):
        self.prep=GrootPrep(model_path,data_config); self.board=IQ9Board(os.path.join(work,"board"))
        self.noise=np.load(noise_bank); self.ep=0; self.call=0; self._staged=None; self.work=work; self.log=[]
        assert self.noise.shape[2:]==(1,16,32),self.noise.shape
        print(f"[M7-IQ9] board loaded {len(self.board.loaded)} graphs ctx_load={self.board.load_us/1e3:.1f}ms; noise {self.noise.shape}",flush=True)
        print(f"[M7-IQ9] ctx set: {os.environ.get('M7_CTX_LIST','(default fp16)')}",flush=True)
        print("[M7-IQ9] SERVER READY",flush=True)
    def get_modality_config(self): return self.prep.dc.modality_config()
    def reset_episode(self,data): self.ep=int(data["episode"]); self.call=0; return {"ok":True,"episode":self.ep}
    def get_action(self, observations):
        t0=time.time(); raws,tinfo=self.prep.obs_to_raws(observations); ids=raws["input_ids"]
        if self._staged is None or not np.array_equal(ids,self._staged):
            self.board.stage_constants(ids,raws["attention_mask"],raws["position_ids"]); self._staged=ids.copy()
        noise=np.ascontiguousarray(self.noise[self.ep,self.call],dtype=np.float32)
        action_norm,tm=self.board.gframe(raws["pixel_values"],raws["state"],noise)
        unnorm=self.prep.unapply(action_norm)
        reply={k:np.ascontiguousarray(np.asarray(v),dtype=np.float32) for k,v in unnorm.items()}
        dt=(time.time()-t0)*1000.0
        self.log.append({"ep":self.ep,"call":self.call,"server_ms":round(dt,2),
                         "board_wall_ms":round(tm["_wall_s"]*1000,1),"board_gframe_ms":round(tm["_gframe_s"]*1000,1),
                         "board_push_ms":round(tm["_push_s"]*1000,1),"wall_us":tm.get("wall_us"),
                         "vit":tm.get("vit"),"llm":tm.get("llm"),"vlln":tm.get("vlln"),"state":tm.get("state"),
                         "aenc":tm.get("aenc"),"dit":tm.get("dit"),"adec_sum":tm.get("adec"),
                         "action_mean_abs":round(float(np.abs(action_norm).mean()),5),"tinfo":tinfo})
        self.call+=1
        if len(self.log)%20==0: print(f"[M7-IQ9] call {len(self.log)} board_wall={tm['_wall_s']*1000:.0f}ms",flush=True)
        return reply
    def dump_log(self):
        p=os.path.join(self.work,"server_calllog.json"); json.dump(self.log,open(p,"w"),indent=1); return p
    def close(self): self.board.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_path",default=_gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain"))
    ap.add_argument("--data_config",default="fourier_gr1_arms_waist")
    ap.add_argument("--noise_bank",required=True); ap.add_argument("--port",type=int,default=5555)
    ap.add_argument("--work",default="/tmp/m7_work_iq9"); args=ap.parse_args()
    os.makedirs(args.work,exist_ok=True)
    from gr00t.eval.robot import RobotInferenceServer
    shim=IQ9Shim(args.model_path,args.work,args.noise_bank,data_config=args.data_config)
    server=RobotInferenceServer(shim,port=args.port); server.register_endpoint("reset_episode",shim.reset_episode)
    try: server.run()
    finally:
        p=shim.dump_log(); print(f"[M7-IQ9] call log -> {p}",flush=True)
        print("[M7-IQ9] health:",shim.board.health(),flush=True); shim.close()

if __name__=="__main__": main()
