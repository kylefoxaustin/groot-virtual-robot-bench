"""Build QNN-friendly graphs from the upstream SONIC encoder/decoder ONNX (policy/release/model_{encoder,decoder}.onnx).
usage: python3 make_graphs.py <dir with model_encoder.onnx + model_decoder.onnx> <out dir>
Why the split: the encoder ends in the FSQ quantizer (Tanh -> scale -> Round). In fp16 on the NPU a value that sits
near a rounding boundary can round to the neighbouring level and flip a token, so the NPU graph stops at the value
before Round and the CPU does the Round (sonic_host.fsq_select; the deploy shim does the same in C++).
sonic_enc_core: obs_dict[1,1762] -> pre_round[3,2,32]  (the three FSQ pre-Round values
  s_k = 15.5155*tanh(z_k+0.0322)-0.5 for encoders k=0 g1, 1 teleop, 2 smpl; all three run every
  tick, exactly like upstream). Host then does: tok = round(s)/16 (FSQ STE forward value) and
  selects encoder k = int(obs_dict[0]) -> encoded_tokens[1,64] (row-major over [2,32]).
sonic_dec: decoder unchanged (renamed copy so DLC basename == graph name).
"""
import onnx, sys
from onnx import helper, utils
src = sys.argv[1]; out = sys.argv[2]
m = onnx.load(f"{src}/model_encoder.onnx")
outs = ["/quantizer/Sub_output_0", "/quantizer_1/Sub_output_0", "/quantizer_2/Sub_output_0"]
e = utils.Extractor(m)
sub = e.extract_model(["obs_dict"], outs)
# reshape each [1,2,1,32]? -> check shapes via inference then concat
g = sub.graph
rs = []
for i, o in enumerate(outs):
    c = helper.make_tensor(f"rs_shape_{i}", onnx.TensorProto.INT64, [3], [1, 2, 32])
    g.initializer.append(c)
    g.node.append(helper.make_node("Reshape", [o, f"rs_shape_{i}"], [f"pre_{i}"]))
    rs.append(f"pre_{i}")
g.node.append(helper.make_node("Concat", rs, ["pre_round"], axis=0))
while len(g.output): g.output.pop()
g.output.append(helper.make_tensor_value_info("pre_round", onnx.TensorProto.FLOAT, [3, 2, 32]))
sub.graph.name = "sonic_enc_core"
onnx.checker.check_model(sub)
onnx.save(sub, f"{out}/sonic_enc_core.onnx")
mf = onnx.load(f"{src}/model_encoder.onnx"); mf.graph.name = "sonic_enc_full"; onnx.save(mf, f"{out}/sonic_enc_full.onnx")
d = onnx.load(f"{src}/model_decoder.onnx"); d.graph.name = "sonic_dec"; onnx.save(d, f"{out}/sonic_dec.onnx")
print("ok", [n.op_type for n in sub.graph.node].count("Gemm"), "gemms in core")
