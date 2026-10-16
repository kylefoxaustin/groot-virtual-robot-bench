#!/usr/bin/env python3
"""Stand-ins for the drone's other on-board AI jobs, with RANDOM weights from a FIXED seed, so every machine runs byte-identical
weights. They exist to load the computer realistically (same graphs, ops and sizes), not to do the jobs.

  seg_512 / seg_1024 : torchvision deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21,
                       aux_loss=False).eval(), wrapped to return ["out"], input "img" [1,3,R,R], output "seg"
                       [1,21,R,R]; torch.onnx.export opset 17, dynamo=False, EVAL, do_constant_folding=True.
                       (the recipe of our first, unseeded stand-in of 2026-09-25; only the seed is new). RANDOM weights.
  radar_fusion       : a PointNet-style late fusion of a 512-point radar cloud, input "pc" [1,5,512] -> "feat" [1,32,512].

Also writes seeded inputs and onnxruntime fp32 reference outputs (for numeric output checks of a compiled stand-in), plus a
manifest with the sha256 of every file. The timed GPU burst feeds ZERO inputs (vision_worker.py); the seeded inputs are for
output checks only.

usage: python3 drone/contention/make_standins.py [OUT_DIR] [SEED]    (default OUT_DIR = $DRONE_STANDINS, SEED = 20261015)
Needs torch, torchvision, onnx, onnxruntime (CPU). Compare the printed MANIFEST sha256 with results/drone/standins_MANIFEST.json:
identical ONNX bytes need the same torch / torchvision (2.9.0 / 0.24.0); the weights depend only on the seed.
"""
import sys, os, json, hashlib, time
import numpy as np, torch, torch.nn as nn, torchvision

OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ["DRONE_STANDINS"]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261015
os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


class Wrap(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)["out"]


N, C = 512, 5
class RadarFusion(nn.Module):   # the structure of the first (2026-09-25) radar stand-in
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(C, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.ReLU())
        self.fuse = nn.Sequential(nn.Conv1d(64 + 128, 128, 1), nn.ReLU(), nn.Conv1d(128, 64, 1), nn.ReLU(), nn.Conv1d(64, 32, 1))
    def forward(self, x):
        local = self.mlp1(x); g = self.mlp2(local)
        gmax = torch.max(g, dim=2, keepdim=True).values
        gexp = gmax.expand(-1, -1, x.shape[2])
        return self.fuse(torch.cat([local, gexp], dim=1))


man = {"seed": SEED, "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "torch": torch.__version__, "torchvision": torchvision.__version__, "files": {}, "notes": {}}

# --- seg: ONE model instance (seed) exported at both resolutions (weights identical across 512/1024) ---
torch.manual_seed(SEED)
seg = Wrap(torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
    weights=None, weights_backbone=None, num_classes=21, aux_loss=False)).eval()
# BN RECALIBRATION (new vs the first stand-in, numerics only): with torchvision's random init and default BN running stats
# (mean 0, var 1) the signal dies inside the backbone; the seeded seg output was input-INDEPENDENT (zero tile vs random
# tile max |diff| 9e-10, one argmax class), so no output check could tell a correct compile from a broken one.
# Re-estimate the BN running stats on seeded ImageNet-normalised noise (cumulative average, batch 2) so every layer
# carries unit-variance signal. Conv weights stay the seeded random init; the exported graph (ops, shapes, MACs) is
# unchanged (BN folds into the convs at export either way). Per-tile time matched the first stand-in on Thor (ratio 1.0007).
_g = torch.Generator().manual_seed(SEED + 7)
_m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); _s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
for mod in seg.modules():
    if isinstance(mod, nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None
seg.train()
with torch.no_grad():
    for _ in range(16): seg((torch.rand(2, 3, 512, 512, generator=_g) - _m) / _s)
seg.eval()
for R in (512, 1024):
    p = f"{OUT}/seg_{R}.onnx"
    with torch.no_grad():
        torch.onnx.export(seg, torch.zeros(1, 3, R, R), p, input_names=["img"], output_names=["seg"], opset_version=17,
                          dynamo=False, training=torch.onnx.TrainingMode.EVAL, do_constant_folding=True)
torch.manual_seed(SEED + 1)
rad = RadarFusion().eval()
with torch.no_grad():
    torch.onnx.export(rad, torch.zeros(1, C, N), f"{OUT}/radar_fusion.onnx", input_names=["pc"], output_names=["feat"],
                      opset_version=17, dynamo=False)

# --- seeded check inputs + fp32 references (onnxruntime CPU, the reference the boards are checked against) ---
import onnxruntime as ort
rng = np.random.default_rng(SEED)
# 4 seg tiles: ImageNet-normalised uniform pixels (a photo-like range), and the zero tile the GPU burst uses
mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
tiles = np.stack([((rng.random((3, 512, 512), dtype=np.float32) - mean) / std) for _ in range(4)])[:, None]   # [4,1,3,512,512]
clouds = rng.standard_normal((8, 1, C, N), dtype=np.float32)                                                 # radar_worker draws randn
so = ort.InferenceSession(f"{OUT}/seg_512.onnx", providers=["CPUExecutionProvider"])
ro = ort.InferenceSession(f"{OUT}/radar_fusion.onnx", providers=["CPUExecutionProvider"])
seg_ref = np.stack([so.run(None, {"img": t})[0] for t in tiles]); seg_zero = so.run(None, {"img": np.zeros((1, 3, 512, 512), np.float32)})[0]
rad_ref = np.stack([ro.run(None, {"pc": c})[0] for c in clouds])
np.savez(f"{OUT}/check_inputs_fp32.npz", seg_tiles=tiles, radar_clouds=clouds)
np.savez(f"{OUT}/check_ref_fp32.npz", seg_out=seg_ref, seg_zero_out=seg_zero, radar_out=rad_ref)
for i, t in enumerate(tiles): t.astype(np.float32).tofile(f"{OUT}/seg_tile{i}.raw")   # NCHW fp32 raw
for i, c in enumerate(clouds): c.astype(np.float32).tofile(f"{OUT}/radar_cloud{i}.raw")

am = seg_ref.argmax(2)
man["notes"] = {
    "seg_out_absmax": float(np.abs(seg_ref).max()), "seg_out_std": float(seg_ref.std()),
    "seg_argmax_classes_used": int(len(np.unique(am))), "seg_argmax_top_class_share": float(np.bincount(am.ravel()).max() / am.size),
    "seg_zero_tile_out_absmax": float(np.abs(seg_zero).max()),
    "radar_out_absmax": float(np.abs(rad_ref).max()), "radar_out_std": float(rad_ref.std()),
    "radar_mflop_per_frame": 2 * N * (C * 64 + 64 * 64 + 64 * 128 + 192 * 128 + 128 * 64 + 64 * 32) / 1e6,
    "provenance": "seg = torchvision deeplabv3_mobilenet_v3_large export recipe of the first (2026-09-25) stand-in; radar = its PointNet-style fusion; both RANDOM conv weights from a fixed seed; seg BN running stats re-estimated on seeded noise so the output depends on the input (default stats gave an input-independent output); graph/ops/shapes unchanged",
}
for f in sorted(os.listdir(OUT)):
    if f != "MANIFEST.json": man["files"][f] = {"sha256": sha(f"{OUT}/{f}"), "bytes": os.path.getsize(f"{OUT}/{f}")}
json.dump(man, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print(json.dumps(man["notes"], indent=1))
