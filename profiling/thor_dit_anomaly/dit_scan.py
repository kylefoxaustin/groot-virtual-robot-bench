import json
d=json.load(open("dit_prof.json"))
rows=[e for e in d if isinstance(e,dict) and e.get("name") and e.get("averageMs") is not None]
import re
def has(sub): return [ (e["name"][:40],round(e["averageMs"],4)) for e in rows if sub in e["name"].lower()]
for kw in ["mha","softmax","gemm","matmul","__mye","fc_","silu","gelu"]:
    L=has(kw); print(f"{kw:8s} count={len(L):3d} time={sum(e[1] for e in L):.4f}ms")
print("\n--- the 'other' bucket members (unclassified) ---")
def cls(n):
    s=n.lower()
    if any(k in s for k in ["gemm","matmul","conv","einsum","fc_","_mm","dense","myelin","foreign","linear","bmm","qkv"]): return "COMPUTE"
    if "layernorm" in s or "rmsnorm" in s or ("norm" in s and "reformat" not in s): return "NORM"
    if "softmax" in s: return "Softmax"
    if any(k in s for k in ["gelu","silu","sigmoid","swish","add","mul","sub","elementwise","pointwise","pwn","bias","act","residual"]): return "elementwise"
    if any(k in s for k in ["reshape","transpose","shuffle","concat","slice","copy","gather","permute","reformat","cast","pad","constant"]): return "layout"
    return "other"
oth=[(e["name"],round(e["averageMs"],4)) for e in rows if cls(e["name"])=="other"]
for nm,t in sorted(oth,key=lambda x:-x[1]):
    print(f"  {t:.4f}ms  {nm[:90]}")
