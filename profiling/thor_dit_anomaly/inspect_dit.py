import json
d=json.load(open("dit_layers.json"))
layers=d if isinstance(d,list) else d.get("Layers",d.get("layers",[]))
dicts=[L for L in layers if isinstance(L,dict)]
print("dict layers:",len(dicts))
if dicts: print("keys:",list(dicts[0].keys()))
for L in dicts:
    nm=L.get("Name","")
    if nm.startswith("__mye131822") or nm.startswith("node_MatMul_1356"):
        print("=== NAME:",nm[:70],"...")
        for k,v in L.items():
            if k=="Name": continue
            print("   ",k,":",str(v)[:200])
        print()
