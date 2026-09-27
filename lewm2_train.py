"""lewm2_train.py — faithful LeWorldModel training loop (AdamW, batch 128)."""
import numpy as np
import torch
from lewm2 import LeWM2, LATENT
from jepa import collapse_metrics

BATCH = 128
EPOCHS = 40          # small data; paper used 10 on larger sets
LR = 1e-3            # optimizer/lr unspecified in paper -> disclosed choice
LAM = 1.0            # paper's 0.1 collapses in this MLP/vector/M=256 setup; tuned up (disclosed)


def main():
    torch.manual_seed(0)
    d = np.load("lewm2_data.npz")
    F, B = d["F"], d["B"]
    obs_mean, obs_std = d["obs_mean"], d["obs_std"]
    Fn = torch.tensor((F - obs_mean) / obs_std, dtype=torch.float32)     # [N,4,25]
    Bt = torch.tensor(B, dtype=torch.float32)                            # [N,3,25]
    print(f"train windows {len(Fn)}")

    model = LeWM2(latent=LATENT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    N = len(Fn)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(N)
        tp = ts = 0.0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 8:      # BatchNorm needs a real batch
                continue
            loss, lp, ls = model.loss(Fn[idx], Bt[idx], lam=LAM)
            opt.zero_grad(); loss.backward(); opt.step()
            tp += lp * len(idx); ts += ls * len(idx)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                z = model.embed(Fn[:4096, 0]); cm = collapse_metrics(z)
                v = (z - z.mean(0)).var(0)
            print(f"ep {ep:02d}  pred {tp/N:.4f}  sigreg {ts/N:.4f}  "
                  f"latent_std {cm['latent_std']:.3f}  eff_rank {cm['eff_rank']:.2f}  "
                  f"var[min/mean/max] {v.min():.2f}/{v.mean():.2f}/{v.max():.2f}")

    torch.save({"model": model.state_dict(), "latent": LATENT,
                "obs_mean": obs_mean, "obs_std": obs_std}, "lewm2_ckpt.pt")
    print("saved -> lewm2_ckpt.pt")


if __name__ == "__main__":
    main()
