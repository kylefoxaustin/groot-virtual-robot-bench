#!/usr/bin/env python3
"""DDR GB/s per GR00T stage on Jetson Thor / Orin, DERIVED from MEASURED BPMP actmon.

actmon (mc_all_avg_activity) is a raw activity counter, not GB/s, and it compresses near saturation.
So each stage's mean actmon is interpolated on the board's MEASURED GPU device-to-device copy
calibration curve (gpu_bw_cal.py: GB/s = 2*bytes/wall, sampled together with actmon), and %-of-peak is
taken against the MEASURED achieved saturation, never the datasheet number.

usage:  derive_gbps.py RECORD.json            # recompute every stage of every board in the record, print, verify
        derive_gbps.py RECORD.json --write    # rewrite the derived fields in place"""
import json, sys
import numpy as np


def curve(cal):
    pts = sorted((p["actmon"], p["gbps"]) for p in cal["points"])
    return np.array([a for a, _ in pts], float), np.array([g for _, g in pts], float)


def gbps(actmon, xs, ys):
    return float(np.interp(actmon, xs, ys))      # clamps at both ends: above saturation -> the saturated value


def derive(board):
    cal = board["gpu_d2d_calibration"]; xs, ys = curve(cal); sat = cal["achieved_saturation"]["gbps"]
    out = {}
    for k, s in board["stages"].items():
        g = gbps(s["actmon"], xs, ys)
        lo, hi = gbps(s["actmon"] - s["sd"], xs, ys), gbps(s["actmon"] + s["sd"], xs, ys)
        out[k] = {"ddr_gbps": round(g, 1), "ddr_gbps_band": [round(lo, 1), round(hi, 1)], "ddr_pct_achieved_sat": round(100 * g / sat, 1)}
    return out


if __name__ == "__main__":
    rec = json.load(open(sys.argv[1])); bad = 0
    for b in ("thor", "orin"):
        for k, v in derive(rec[b]).items():
            pub = rec[b]["stages"][k]
            ok = abs(v["ddr_gbps"] - pub["ddr_gbps"]) <= 0.15 and abs(v["ddr_pct_achieved_sat"] - pub["ddr_pct_achieved_sat"]) <= 0.15 \
                and all(abs(x - y) <= 0.15 for x, y in zip(v["ddr_gbps_band"], pub["ddr_gbps_band"]))
            bad += not ok
            print(f"{b:5s} {k:4s} actmon {pub['actmon']:>10,}  -> {v['ddr_gbps']:6.1f} GB/s {v['ddr_gbps_band']}  "
                  f"{v['ddr_pct_achieved_sat']:5.1f}% of achieved   published {pub['ddr_gbps']} {pub['ddr_gbps_band']} {pub['ddr_pct_achieved_sat']}%"
                  + ("" if ok else "   <-- MISMATCH"))
            if "--write" in sys.argv: pub.update(v)
    if "--write" in sys.argv: json.dump(rec, open(sys.argv[1], "w"), indent=1)
    sys.exit(1 if bad else 0)
