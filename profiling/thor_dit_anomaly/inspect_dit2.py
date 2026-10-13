import json
d=json.load(open("dit_layers.json"))
print("top type:",type(d).__name__)
if isinstance(d,dict): print("top keys:",list(d.keys()))
layers=d.get("Layers") if isinstance(d,dict) else d
print("layers type:",type(layers).__name__,"len:",len(layers) if hasattr(layers,"__len__") else "?")
print("first elem type:",type(layers[0]).__name__)
print("first elem:",str(layers[0])[:400])
