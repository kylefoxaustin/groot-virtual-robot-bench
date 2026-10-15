#!/usr/bin/env python3
"""IQ-9075 CPU control arm (Cortex-A78, onnxruntime CPU EP, fp32) for the SONIC encoder+decoder.
Runs the UPSTREAM ONNX unchanged (encoder incl. its own Round/FSQ + mode select).
Usage (on the board):  $IQ9_PY_ROOT/bin/python3 -I cpu_ort_run.py --mode correct|latency
   --enc model_encoder.onnx --dec model_decoder.onnx --enc_in enc_in.f32 --dec_in dec_in.f32 --out DIR
   [--threads 4] [--reps 2000 --warmup 50 --period_us 20000] [--n N]
Outputs mirror sonic_bench: correct -> enc_raw.f32 [N,64 tokens], dec_raw.f32 [N,29];
latency -> lat.csv (rep,i,enc_us,dec_us,glue_us,step_us,late_us) + lat_actions.f32.
Latency loop: enc(enc_in[i]) -> tokens written into dec_in[i][0:64] -> dec -> action (same as NPU loop).
"""
import argparse, os, time
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="correct"); ap.add_argument("--enc"); ap.add_argument("--dec")
    ap.add_argument("--enc_in"); ap.add_argument("--dec_in"); ap.add_argument("--out", default=".")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--period_us", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    E = ort.InferenceSession(a.enc, so, providers=["CPUExecutionProvider"])
    D = ort.InferenceSession(a.dec, so, providers=["CPUExecutionProvider"])
    ein = np.fromfile(a.enc_in, np.float32).reshape(-1, 1762)
    din = np.fromfile(a.dec_in, np.float32).reshape(-1, 994)
    N = len(ein) if a.n == 0 else min(a.n, len(ein))
    assert len(din) >= N
    if a.mode == "correct":
        tok = np.concatenate([E.run(None, {"obs_dict": ein[i:i + 1]})[0] for i in range(N)])
        act = np.concatenate([D.run(None, {"obs_dict": din[i:i + 1]})[0] for i in range(N)])
        tok.astype(np.float32).tofile(f"{a.out}/enc_raw.f32"); act.astype(np.float32).tofile(f"{a.out}/dec_raw.f32")
        print("CORRECT_DONE N=%d threads=%d ort=%s" % (N, a.threads, ort.__version__))
        return
    rows, acts = [], np.zeros((a.reps, 29), np.float32)
    nxt = time.perf_counter_ns()
    pc = time.perf_counter_ns
    for r in range(-a.warmup, a.reps):
        i = r % N
        late = 0
        if a.period_us > 0:
            n0 = pc()
            if n0 < nxt:
                time.sleep((nxt - n0) / 1e9)
            else:
                late = (n0 - nxt) // 1000
            nxt += a.period_us * 1000
        s0 = pc()
        x = ein[i:i + 1]
        a0 = pc(); t = E.run(None, {"obs_dict": x})[0]; a1 = pc()
        d = din[i:i + 1].copy(); d[0, :64] = t[0]
        b0 = pc(); y = D.run(None, {"obs_dict": d})[0]; b1 = pc()
        s1 = pc()
        if r >= 0:
            acts[r] = y[0]
            ea, db, st = (a1 - a0) // 1000, (b1 - b0) // 1000, (s1 - s0) // 1000
            rows.append((r, i, ea, db, st - ea - db, st, late))
    with open(f"{a.out}/lat.csv", "w") as f:
        f.write("rep,i,enc_us,dec_us,glue_us,step_us,late_us\n")
        for row in rows:
            f.write(",".join(str(int(v)) for v in row) + "\n")
    acts.tofile(f"{a.out}/lat_actions.f32")
    print("LATENCY_DONE reps=%d warm=%d period_us=%d threads=%d" % (a.reps, a.warmup, a.period_us, a.threads))


if __name__ == "__main__":
    main()
