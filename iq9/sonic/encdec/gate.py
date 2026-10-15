#!/usr/bin/env python3
"""Correctness gate: board (NPU or CPU) outputs vs the host fp32 reference, judged against the fp16
prediction (fp16_predict.py output, written before the board run). Thresholds are derived from the prediction below.

Usage: python3 gate.py <pred.json> <inputs.npz> <board_out_dir> <enc_kind core|full|tokens> [--label X]
board_out_dir must contain enc_raw.f32 and dec_raw.f32 (from sonic_bench --mode correct, or cpu_ort_run.py).
enc_kind: core -> enc_raw is pre_round [N,3,2,32] (host applies Round/16 + mode select, same as the loop);
          full|tokens -> enc_raw is encoded_tokens [N,64].
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sonic_host import fsq_select
from fp16_predict import enc_metrics, dec_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("npz"); ap.add_argument("bdir"); ap.add_argument("enc_kind")
    ap.add_argument("--label", default="")
    ap.add_argument("--arm", default="npu", choices=["npu", "cpu"], help="cpu = fp32 arm: fp16_signature check not applied")
    a = ap.parse_args()
    P = json.load(open(a.pred)); R = np.load(a.pred.replace(".json", "_ref.npz"))
    z = np.load(a.npz); N = P["N"]
    enc_in = z["enc_in"][:N].astype(np.float32)
    raw = np.fromfile(f"{a.bdir}/enc_raw.f32", np.float32)
    nb = len(raw) // (192 if a.enc_kind == "core" else 64)
    N = min(N, nb)
    if a.enc_kind == "core":
        tok = fsq_select(raw[:N * 192].reshape(N, 3, 2, 32), enc_in[:N])
    else:
        tok = raw[:N * 64].reshape(N, 64)
    act = np.fromfile(f"{a.bdir}/dec_raw.f32", np.float32)[:N * 29].reshape(N, 29)
    ref_tok, ref_act = R["ref_tok"][:N], R["ref_act"][:N]
    em, dm = enc_metrics(tok, ref_tok), dec_metrics(act, ref_act)
    acc = P["arms"]["f16acc"]; sto = P["arms"]["f16store"]
    pe, pd = acc["enc"], acc["dec"]
    thr = {
        "enc_token_exact_frac_min": pe["token_exact_frac"] - max(0.002, 0.5 * (1 - pe["token_exact_frac"])),
        "enc_tick_anyflip_rate_max": max(2 * pe["tick_anyflip_rate"], pe["tick_anyflip_rate"] + 0.02),
        "dec_rel_mae_max": 2 * pd["rel_mae"],
        "dec_max_abs_rad_max": 2 * pd["max_abs_rad"],
        "dec_cos_min": 0.999,
        "dec_rel_mae_min_fp16_signature": 0.1 * sto["dec"]["rel_mae"],
    }
    checks = {
        "enc_token_exact_frac": em["token_exact_frac"] >= thr["enc_token_exact_frac_min"],
        "enc_tick_anyflip_rate": em["tick_anyflip_rate"] <= thr["enc_tick_anyflip_rate_max"],
        "dec_rel_mae": dm["rel_mae"] <= thr["dec_rel_mae_max"],
        "dec_max_abs_rad": dm["max_abs_rad"] <= thr["dec_max_abs_rad_max"],
        "dec_cos": dm["cos"] >= thr["dec_cos_min"],
        "fp16_signature": dm["rel_mae"] >= thr["dec_rel_mae_min_fp16_signature"],
    }
    out = {"label": a.label or a.bdir, "N": N, "enc_kind": a.enc_kind, "board_enc": em, "board_dec": dm,
           "pred_f16store": sto, "pred_f16acc": acc, "thresholds": thr, "checks": checks,
           "arm": a.arm,
           "verdict": "PASS" if all(v for k, v in checks.items() if not (a.arm == "cpu" and k == "fp16_signature")) else "FAIL",
           "note_fp16_signature": "only meaningful for an NPU fp16 arm; a CPU fp32 arm is EXPECTED to fail it (that is the point)"}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{a.bdir}/gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
