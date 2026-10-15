"""Summarise lat.csv (us) -> p50/p99/max per column + compute misses at the 20 ms tick. Usage: lat_summary.py DIR [DIR..]"""
import sys, json, numpy as np
for d in sys.argv[1:]:
    a = np.genfromtxt(f"{d}/lat.csv", delimiter=",", names=True)
    out = {"dir": d, "reps": int(len(a))}
    for c in [c for c in ["enc_us", "dec_us", "glue_us", "step_us", "late_us", "enc_acc_us", "dec_acc_us"] if c in a.dtype.names]:
        v = a[c]; out[c] = {"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
                            "max": float(v.max()), "mean": float(v.mean()), "min": float(v.min())}
    out["deadline_miss_20ms"] = int((a["step_us"] > 20000).sum())
    json.dump(out, open(f"{d}/lat_summary.json", "w"), indent=1)
    print(d, "reps", out["reps"], " ".join(f"{c}={out[c]['p50']:.0f}/{out[c]['p99']:.0f}/{out[c]['max']:.0f}" for c in ["enc_us", "dec_us", "glue_us", "step_us", "enc_acc_us", "dec_acc_us"] if c in out), "miss20ms", out["deadline_miss_20ms"])
