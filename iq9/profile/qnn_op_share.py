#!/usr/bin/env python3
"""IQ-9075 per-op "matmul vs glue" share of a GR00T stage, from QNN HTP detailed profiling.

Method:
  1. on the board, run the stage's fp16 context binary with detailed profiling:
       qnn-net-run --backend libQnnHtp.so --retrieve_context groot_<stage>_fp16.bin --input_list <list> \
                   --profiling_level detailed --output_dir out_<stage>
  2. qnn-profile-viewer --input_log out_<stage>/qnn-profiling-data_0.log > <stage>_profile.txt
  3. python qnn_op_share.py <stage>_profile.txt
QNN runs every op as its own HTP node (no cross-op fusion), so per-node accelerator cycles are a clean,
unfused anatomy: matmul-class nodes (Conv/MatMul/FullyConnected/Gemm/BatchMatMul) vs everything else
(norms, softmax, elementwise, layout/reshape/transpose/gather/cast = "glue").

STATUS: re-implementation of a parse step we originally did interactively; not re-run for this release.
qnn-profile-viewer's text layout varies by QAIRT release; if your viewer prints differently, adjust NODE_RE."""
import collections, re, sys

MATMUL = ("conv", "matmul", "fullyconnected", "fully_connected", "gemm", "batchmatmul", "linear", "_fc", "fc_")
# a per-node line: '<node name> [optional (OpType)] : <value> cycles|us'
NODE_RE = re.compile(r"^\s*(?P<name>[^:]+?)\s*(?:\((?P<type>[A-Za-z0-9_]+)\))?\s*:\s*(?P<val>[0-9][0-9.,]*)\s*(?P<unit>cycles|us)\b", re.I)
SKIP = ("accelerator", "backend", "netrun", "rpc", "qnn ", "total", "graph", "execute", "finalize", "init", "deinit")


def kind(name, typ):
    s = f"{typ or ''} {name}".lower()
    return "matmul" if any(k in s for k in MATMUL) else "glue"


def main(path):
    agg = collections.Counter(); n = 0
    for ln in open(path, errors="ignore"):
        m = NODE_RE.match(ln)
        if not m or m.group("name").strip().lower().startswith(SKIP): continue
        v = float(m.group("val").replace(",", "")); agg[kind(m.group("name"), m.group("type"))] += v; n += 1
    tot = sum(agg.values())
    if not n or not tot: sys.exit("no per-node rows found -- was --profiling_level detailed used? adjust NODE_RE for your viewer")
    print(f"nodes={n}  matmul {100 * agg['matmul'] / tot:.1f}%  glue {100 * agg['glue'] / tot:.1f}%  (of summed per-node accelerator time)")


if __name__ == "__main__":
    main(sys.argv[1])
