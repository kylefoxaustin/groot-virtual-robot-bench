#!/usr/bin/env python3
"""Train the drone action head on frozen GR00T N1.5 features (behaviour cloning of the scripted teacher).

1. features: for every frame of drone_nav.npz, the 2048-d pooled GR00T feature (drone/policy.py backend 'eager': bf16
   Gr00tPolicy.get_action with the frame as the GR1 'ego_view' camera and the instruction as the task; the backbone output is
   pooled after the action head's vlln + vl_self_attention) -> feats.npz (cached).
2. head: MLP (2048 + 4 state) -> 512 -> 512 -> 16 x 4, GELU + LayerNorm, AdamW 1e-3, cosine schedule, 3,000 steps of batch
   256, 90/10 split, inputs and targets z-scored with statistics over ALL frames.
3. export: drone_head.pt (torch) and drone_head.npz (numpy, what the benchmark loads).

  python3 drone/train/train_head.py DATASET_DIR      (needs the GR00T env + a GPU; ~15 min on an RTX 5090)
The split and the minibatches are not seeded (they were not in our run), so a retrained head differs from
drone/assets/drone_head.npz, and so will its closed-loop numbers. Use the shipped head to compare with ours.
"""
import os, sys, time, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import torch, torch.nn as nn
import cv2

D = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
DEV = "cuda"


def extract_features():
    cache = f"{D}/feats.npz"
    if os.path.exists(cache):
        z = np.load(cache); print("features cached", z["feats"].shape); return z["feats"], z["states"], z["chunks"]
    import policy as P
    pol = P.DronePolicy("eager", head=os.path.join(HERE, "..", "assets", "drone_head.npz"))   # the head is unused here
    d = np.load(f"{D}/drone_nav.npz", allow_pickle=True)
    imgs, states, chunks, instrs = d["images"], d["states"], d["action_chunks"], d["instructions"]
    feats = np.zeros((len(imgs), 2048), np.float32); t0 = time.time()
    for i in range(len(imgs)):
        feats[i] = pol.feature(imgs[i], str(instrs[i]))
        if i % 500 == 0: print(f"  feature {i}/{len(imgs)}  {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(cache, feats=feats, states=states, chunks=chunks)
    return feats, states, chunks


class Head(nn.Module):
    def __init__(self, fin=2048, sin=4, H=16, A=4):
        super().__init__(); self.H, self.A = H, A
        self.net = nn.Sequential(nn.Linear(fin + sin, 512), nn.GELU(), nn.LayerNorm(512),
                                 nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, H * A))

    def forward(self, f, s): return self.net(torch.cat([f, s], -1)).view(-1, self.H, self.A)


def train(feats, states, chunks):
    f = torch.tensor(feats, device=DEV); s = torch.tensor(states, device=DEV); y = torch.tensor(chunks, device=DEV)
    fmu, fsd = f.mean(0, keepdim=True), f.std(0, keepdim=True) + 1e-6; f = (f - fmu) / fsd
    smu, ssd = s.mean(0, keepdim=True), s.std(0, keepdim=True) + 1e-6; s = (s - smu) / ssd
    ymu, ysd = y.reshape(-1, 4).mean(0), y.reshape(-1, 4).std(0) + 1e-6; yn = (y - ymu) / ysd
    N = len(f); idx = torch.randperm(N); tr, va = idx[:int(.9 * N)], idx[int(.9 * N):]
    net = Head().to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000); best = 1e9
    for ep in range(3000):
        net.train(); b = tr[torch.randint(0, len(tr), (256,))]
        loss = ((net(f[b], s[b]) - yn[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if ep % 500 == 0 or ep == 2999:
            net.eval()
            with torch.no_grad(): vl = ((net(f[va], s[va]) - yn[va]) ** 2).mean().item()
            print(f"  step {ep} train {loss.item():.4f} val {vl:.4f}", flush=True); best = min(best, vl)
    return net, best


if __name__ == "__main__":
    feats, states, chunks = extract_features()
    net, best = train(feats, states, chunks)
    torch.save({"net": net.state_dict()}, f"{D}/drone_head.pt")
    import drone_head_np
    drone_head_np.export(f"{D}/drone_head.pt", f"{D}/feats.npz", f"{D}/drone_head.npz")
    print(json.dumps({"frames": int(len(feats)), "val_mse_normalised": round(best, 4), "head": f"{D}/drone_head.npz"}))
