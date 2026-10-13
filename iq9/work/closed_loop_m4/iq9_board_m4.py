#!/usr/bin/env python3
"""Board transport: drive the iq9 resident GFRAME loop from the host.

Launches the EXISTING board resident_orch (7 GR00T v73 fp16 ctx bins, DDR-resident)
over a persistent ssh pipe and speaks its verified line protocol (GFRAME/QUIT). An
ssh ControlMaster socket makes per-step scp cheap. Constant raws (input_ids,
attention_mask, position_ids, time_proj_t*) are staged ONCE; only the per-step
varying raws (pixel_values, state, init_actions) are pushed each step.

Board safety: clean QUIT only (never SIGKILL); a killed resident_orch can wedge the Hexagon DSP
on kill -9. close() sends QUIT and waits.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
_BQ = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_QNN_ROOT", "/opt/qnn"))
_BP = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_PY_ROOT", "/opt/iq9-python"))
import os, sys, time, subprocess, tempfile
import numpy as np

BOARD = os.environ.get("IQ9_HOST", "iq9")
QNN = _BQ+""
BROOT = _B+"/optionB"
BLAS = _BP+"/python/lib/python3.12/site-packages/torch/lib/libopenblas.so.0"
CTX = f"{BROOT}/ctx_ft"                 # M4: fine-tuned bins
GLUE = f"{BROOT}/glue_ft"              # M4: FT glue (mlp1/future/posembed) + reused embed
SRC_IO = f"{BROOT}/io_ft"              # M4: FT time_proj_t*.raw
IO_M2 = f"{BROOT}/io_m4"               # M4 dedicated per-step dir
CTX_BINS = ["groot_vit_fp16.bin", "groot_llm_fp16.bin", "vlln_vl_self_attention_fp16.bin",
            "state_encoder_fp16.bin", "action_encoder_fp16.bin", "groot_dit_fp16.bin",
            "action_decoder_fp16.bin"]
PROTO = ("LOADED", "READY", "OK", "OKGFRAME", "PONG", "BYE", "ERR")


class IQ9Board:
    def __init__(self, workdir, blas_threads=6, log_level="error"):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        # unix socket path must be < 108 bytes -> keep it short (not under workdir)
        self.ctl = f"/tmp/.m2ssh_{os.getpid()}.sock"
        self._ssh_base = ["ssh", "-o", "ControlMaster=auto", "-o", f"ControlPath={self.ctl}",
                          "-o", "ControlPersist=300", "-o", "ServerAliveInterval=30"]
        # open master connection
        subprocess.run(self._ssh_base + ["-fN", BOARD], check=True)
        self._setup_board_io()
        self._launch_resident(blas_threads, log_level)

    def _ssh(self, cmd, **kw):
        return subprocess.run(self._ssh_base + [BOARD, cmd], **kw)

    def _scp(self, local_files, remote_dir):
        # reuse master connection for speed
        cmd = ["scp", "-o", f"ControlPath={self.ctl}", "-q"] + local_files + \
              [f"{BOARD}:{remote_dir}/"]
        subprocess.run(cmd, check=True)

    def _setup_board_io(self):
        # fresh io_m2 with the fixed time_proj_* copied from the existing io dir
        self._ssh(f"mkdir -p {IO_M2} && cp -f {SRC_IO}/time_proj_t*.raw {IO_M2}/", check=True)

    def stage_constants(self, input_ids_i64, attention_mask_f32, position_ids_f32):
        """Push the per-episode/constant raws once."""
        d = tempfile.mkdtemp(dir=self.workdir)
        np.ascontiguousarray(input_ids_i64, np.int64).tofile(f"{d}/input_ids.raw")
        np.ascontiguousarray(attention_mask_f32, np.float32).tofile(f"{d}/attention_mask.raw")
        np.ascontiguousarray(position_ids_f32, np.float32).tofile(f"{d}/position_ids.raw")
        self._scp([f"{d}/input_ids.raw", f"{d}/attention_mask.raw", f"{d}/position_ids.raw"], IO_M2)

    def _launch_resident(self, blas_threads, log_level):
        args = [f"{BROOT}/resident/resident_orch", "--backend", f"{QNN}/lib/libQnnHtp.so",
                "--system_library", f"{QNN}/lib/libQnnSystem.so", "--log_level", log_level,
                "--blas", BLAS, "--glue", GLUE, "--blas_threads", str(blas_threads)]
        for b in CTX_BINS:
            args += ["--ctx", f"{CTX}/{b}"]
        remote = (f"cd {BROOT}/resident && LD_LIBRARY_PATH={QNN}/lib "
                  f"ADSP_LIBRARY_PATH={QNN}/skel " + " ".join(args))
        self.errlog = open(os.path.join(self.workdir, "orch_stderr.log"), "w")
        self.p = subprocess.Popen(self._ssh_base + [BOARD, remote],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errlog, text=True, bufsize=1)
        self.loaded, self.load_us = [], None
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("resident_orch died during load; see orch_stderr.log")
            line = line.strip()
            if line.startswith("LOADED"):
                self.loaded.append(line.split()[1])
            elif line.startswith("READY"):
                for t in line.split():
                    if t.startswith("load_us="):
                        self.load_us = int(t.split("=")[1])
                break
            elif line.startswith("ERR"):
                raise RuntimeError(f"load error: {line}")
        self.load_wall = time.time() - t0

    def _readline_proto(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("resident_orch closed stdout; see orch_stderr.log")
            line = line.strip()
            if line.startswith(PROTO):
                return line

    def gframe(self, pixel_f32, state_f32, init_actions_f32):
        """Push the 3 per-step raws, run one GFRAME, return (action(1,16,32), timings, wall_s)."""
        d = tempfile.mkdtemp(dir=self.workdir)
        np.ascontiguousarray(pixel_f32, np.float32).tofile(f"{d}/pixel_values.raw")
        np.ascontiguousarray(state_f32, np.float32).tofile(f"{d}/state.raw")
        np.ascontiguousarray(init_actions_f32, np.float32).tofile(f"{d}/init_actions.raw")
        t0 = time.time()
        self._scp([f"{d}/pixel_values.raw", f"{d}/state.raw", f"{d}/init_actions.raw"], IO_M2)
        push_s = time.time() - t0
        t1 = time.time()
        self.p.stdin.write(f"GFRAME {IO_M2} {IO_M2}/action.raw\n")
        self.p.stdin.flush()
        resp = self._readline_proto()
        if not resp.startswith("OKGFRAME"):
            raise RuntimeError(f"GFRAME failed: {resp} (see orch_stderr.log)")
        gframe_s = time.time() - t1
        tm = {}
        for tok in resp.split()[1:]:
            k, v = tok.split("="); tm[k] = int(v)
        # pull action back
        lf = os.path.join(d, "action.raw")
        subprocess.run(["scp", "-o", f"ControlPath={self.ctl}", "-q",
                        f"{BOARD}:{IO_M2}/action.raw", lf], check=True)
        action = np.fromfile(lf, np.float32).reshape(1, 16, 32)
        wall_s = time.time() - t0
        tm["_push_s"] = push_s; tm["_gframe_s"] = gframe_s; tm["_wall_s"] = wall_s
        return action, tm

    def health(self):
        r = self._ssh("for f in /sys/class/remoteproc/remoteproc*/state; do cat $f; done; "
                      "ps -ef | grep -E 'resident_orch|qnn-net-run' | grep -v grep | wc -l",
                      capture_output=True, text=True)
        return r.stdout.strip()

    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            try:
                self.p.terminate(); self.p.wait(timeout=10)
            except Exception:
                pass
        try:
            self.errlog.close()
        except Exception:
            pass
        # close ssh master
        subprocess.run(self._ssh_base + ["-O", "exit", BOARD],
                       capture_output=True)
