import json,sys,collections
def cls(n):
    s=n.lower()
    if any(k in s for k in ["gemm","matmul","conv","einsum","fc_","_mm","dense","myelin","foreign","linear","bmm","qkv"]): return "COMPUTE"
    if "layernorm" in s or "rmsnorm" in s or ("norm" in s and "reformat" not in s): return "LayerNorm"
    if "softmax" in s: return "Softmax"
    if any(k in s for k in ["gelu","silu","sigmoid","swish","add","mul","sub","elementwise","pointwise","pwn","bias","act","residual"]): return "elementwise"
    if any(k in s for k in ["reshape","transpose","shuffle","concat","slice","copy","gather","permute","reformat","cast","pad","constant"]): return "layout"
    return "other"
d=json.load(open(sys.argv[1]))
rows=d if isinstance(d,list) else d.get("layers",d)
cat=collections.Counter(); n=0; items=[]
for e in rows:
    if not isinstance(e,dict): continue
    nm=e.get("name"); t=e.get("averageMs", e.get("timeMs"))
    if nm and t is not None: cat[cls(nm)]+=t; n+=1; items.append((t,nm,cls(nm)))
tot=sum(cat.values()) or 1e-9
for k,v in cat.most_common(): print(f"{k:12s}{100*v/tot:5.1f}%  {v:.4f}ms")
comp=cat["COMPUTE"]
print(f">>> COMPUTE {100*comp/tot:.1f}%  NON-COMPUTE {100*(tot-comp)/tot:.1f}%  total/inf={tot:.4f}ms  layers={n}")
print("--- TOP 6 LAYERS BY TIME ---")
for t,nm,c in sorted(items,reverse=True)[:6]:
    print(f"{t:.4f}ms  [{c}]  {nm}")
