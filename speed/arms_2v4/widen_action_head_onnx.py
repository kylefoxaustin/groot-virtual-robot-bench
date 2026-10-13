#!/usr/bin/env python3
"""Make the SYNTHETIC 4-arm action head: widen GR00T N1.5's action_encoder / action_decoder ONNX
from action_dim 32 (two GR1 arms + hands, padded) to 64 by ONNX surgery -- the action-dim slices of
the CategorySpecificLinear weights are duplicated. This is a TIMING model only: it is not a trained
4-arm policy and makes no accuracy claim.

  action_encoder: input actions [B,16,32] -> [B,16,64]; the per-embodiment input projection
                  W [32 emb, 32, 1536] -> [32 emb, 64, 1536]
  action_decoder: output [B,49,32] -> [B,49,64]; W [32 emb, 1024, 32] -> [32 emb, 1024, 64] and
                  bias [32 emb, 32] -> [32 emb, 64]

usage: $GROOT_PY speed/arms_2v4/widen_action_head_onnx.py ONNX_DIR      (reads/writes ONNX_DIR/action_head/*.onnx)
writes action_encoder_4arm.onnx and action_decoder_4arm.onnx next to the originals."""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper

A, A2 = 32, 64


def widen(src, dst, io_name, io_axis, rules):
    m = onnx.load(src)                       # loads external data too
    hit = 0
    for i, init in enumerate(m.graph.initializer):
        arr = numpy_helper.to_array(init)
        for shape, axis in rules:
            if tuple(arr.shape) == shape:
                new = np.concatenate([arr, arr], axis=axis)
                m.graph.initializer[i].CopyFrom(numpy_helper.from_array(new, init.name))
                hit += 1
                break
    assert hit == len(rules), f"{src}: expected {len(rules)} action-dim initializers, widened {hit}"
    for v in list(m.graph.input) + list(m.graph.output):
        if v.name == io_name:
            d = v.type.tensor_type.shape.dim[io_axis]
            assert d.dim_value == A, (v.name, d)
            d.dim_value = A2
    del m.graph.value_info[:]                # stale intermediate shapes; TensorRT re-infers
    onnx.save(m, dst, save_as_external_data=True, location=os.path.basename(dst) + ".data")
    onnx.checker.check_model(dst)
    print("wrote", dst)


if __name__ == "__main__":
    ah = os.path.join(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__), "action_head")
    widen(f"{ah}/action_encoder.onnx", f"{ah}/action_encoder_4arm.onnx", "actions", 2,
          [((32, A, 1536), 1)])
    widen(f"{ah}/action_decoder.onnx", f"{ah}/action_decoder_4arm.onnx", "output", 2,
          [((32, 1024, A), 2), ((32, A), 1)])
