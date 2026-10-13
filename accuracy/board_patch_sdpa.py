#!/usr/bin/env python3
"""Patch (and restore) a board's Isaac-GR00T (N1.5) eagle backbone to use torch SDPA
attention instead of flash_attention_2, so GR00T's full torch policy loads on a
Jetson that has no flash_attn. SDPA is exact attention (same math as flash); the
kernel/accumulation-order difference is far below the fp16/fp8 precision effect
the closed-loop comparison measures, and BOTH boards use sdpa consistently.

Edits (with .bak backups, so --restore is exact):
  * eagle2_hg_model/config.json      : every *_attn_implementation -> "sdpa"
  * modeling_eagle2_5_vl.py          : the hardcoded siglip "flash_attention_2"
                                       -> "sdpa"; relax the Qwen2 flash assert.

Usage:
  board_patch.py --isaac <dir> apply
  board_patch.py --isaac <dir> restore
  board_patch.py --isaac <dir> verify   # print current attn settings
"""
import argparse, json, os, shutil, sys

EAGLE = "gr00t/model/backbone/eagle2_hg_model"


def _set_attn(obj):
    n = 0
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k.endswith("attn_implementation"):
                if obj[k] != "sdpa":
                    obj[k] = "sdpa"; n += 1
            else:
                n += _set_attn(obj[k])
    elif isinstance(obj, list):
        for v in obj:
            n += _set_attn(v)
    return n


def apply(isaac):
    cfg = os.path.join(isaac, EAGLE, "config.json")
    mdl = os.path.join(isaac, EAGLE, "modeling_eagle2_5_vl.py")
    for f in (cfg, mdl):
        if not os.path.exists(f + ".bak"):
            shutil.copy2(f, f + ".bak")
    c = json.load(open(cfg))
    # force top-level too (some builds read it directly)
    if c.get("_attn_implementation") != "sdpa":
        c["_attn_implementation"] = "sdpa"
    n = _set_attn(c)
    json.dump(c, open(cfg, "w"), indent=2)
    src = open(mdl).read()
    orig = src
    src = src.replace('config.vision_config._attn_implementation = "flash_attention_2"',
                      'config.vision_config._attn_implementation = "sdpa"')
    # relax the Qwen2 assert: accept sdpa
    src = src.replace(
        'assert (\n                    config.text_config._attn_implementation == "flash_attention_2"\n                ), f"Qwen2 must use flash_attention_2 but got {config.text_config._attn_implementation}"',
        'assert config.text_config._attn_implementation in ("flash_attention_2", "sdpa", "eager"), config.text_config._attn_implementation')
    open(mdl, "w").write(src)
    print(f"[patch] config.json _attn keys set to sdpa: {n}")
    print(f"[patch] modeling_eagle2_5_vl.py changed: {src != orig}")


def restore(isaac):
    for rel in ("config.json", "modeling_eagle2_5_vl.py"):
        f = os.path.join(isaac, EAGLE, rel)
        if os.path.exists(f + ".bak"):
            shutil.copy2(f + ".bak", f)
            print(f"[restore] {rel} <- .bak")
        else:
            print(f"[restore] NO .bak for {rel} (left as-is)")


def verify(isaac):
    cfg = os.path.join(isaac, EAGLE, "config.json")
    c = json.load(open(cfg))
    def show(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith("attn_implementation"):
                    print(f"  {p}{k} = {v}")
                elif isinstance(v, (dict, list)):
                    show(v, p + k + ".")
    show(c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True)
    ap.add_argument("action", choices=["apply", "restore", "verify"])
    a = ap.parse_args()
    {"apply": apply, "restore": restore, "verify": verify}[a.action](a.isaac)
