import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lewm2 import LeWM2, N_HIST
from lewm2_data import onehot_block
from jepa import collapse_metrics

ck = torch.load("lewm2_ckpt.pt", map_location="cpu", weights_only=False)
m = LeWM2(latent=ck["latent"]); m.load_state_dict(ck["model"]); m.eval()
mean, std = ck["obs_mean"], ck["obs_std"]
d = np.load("lewm2_data.npz"); F4 = d["Fte"]                     # [N,4,25]
Fn = torch.tensor((F4 - mean) / std, dtype=torch.float32)
B = len(Fn)
with torch.no_grad():
    Z = m.embed(Fn.reshape(B * 4, -1)).numpy()                  # [N*4, 64]

mu = Z.mean(0); cov = np.cov(Z, rowvar=False); dvar = np.diag(cov)
off = np.sqrt((cov[~np.eye(len(cov), dtype=bool)] ** 2).mean())
cm = collapse_metrics(torch.tensor(Z))
print("=== isotropy of the embedding (saved lewm2 model) ===")
print(f"  latent_std {cm['latent_std']:.3f}  |mean| {np.abs(mu).mean():.3f}  "
      f"var[min/mean/max] {dvar.min():.2f}/{dvar.mean():.2f}/{dvar.max():.2f}  "
      f"off-diag {off:.3f}  eff_rank {cm['eff_rank']:.1f}/{Z.shape[1]}")
rng = np.random.default_rng(0)
V = rng.normal(size=(Z.shape[1], 200)); V /= np.linalg.norm(V, axis=0, keepdims=True)
Pn = ((Z - mu) @ V); Pn = (Pn - Pn.mean(0)) / (Pn.std(0) + 1e-8)
print(f"  random-projection skew {np.abs((Pn**3).mean(0)).mean():.3f}  exkurt {np.abs((Pn**4).mean(0)-3).mean():.3f}")

# action causality: same states, FASTER vs SLOWER -> decoded next ego speed
with torch.no_grad():
    z = m.embed(Fn.reshape(B * 4, -1)).reshape(B, 4, -1)
    hist = z[:, :N_HIST]
    def blk(a): return torch.tensor(np.tile(onehot_block([a] * 5), (B, 1)), dtype=torch.float32)
    zf = m.predictor(hist, blk(3)).numpy()      # FASTER
    zs = m.predictor(hist, blk(4)).numpy()      # SLOWER
# probe: embedding -> ego speed (obs col 3), fit on real frame embeddings
Ze = z.reshape(B * 4, -1).numpy(); yspeed = F4.reshape(B * 4, 25)[:, 3]
Xb = np.concatenate([Ze, np.ones((len(Ze), 1))], 1)
w = np.linalg.lstsq(Xb, yspeed, rcond=None)[0]
def dec(zz): return np.concatenate([zz, np.ones((len(zz), 1))], 1) @ w
vf, vs = dec(zf), dec(zs)
print(f"\n=== action causality: mean decoded next ego-speed  FASTER {vf.mean():.3f}  SLOWER {vs.mean():.3f} ===")

fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
ax[0].hist(dvar, bins=24, color="#2E6F4E", alpha=.85); ax[0].axvline(1, color="k", ls="--")
ax[0].set_title(f"Per-dim variance (want ~1)\nmean={dvar.mean():.2f}"); ax[0].set_xlabel("variance")
im = ax[1].imshow(np.abs(cov), cmap="magma", vmax=np.percentile(np.abs(cov), 99))
ax[1].set_title(f"|covariance| (want diagonal)\noff-diag RMS={off:.3f}"); plt.colorbar(im, ax=ax[1], fraction=.046)
ax[2].hist(Pn[:, 0], bins=60, density=True, color="#2f5aa8", alpha=.7, label="random projection")
xs = np.linspace(-4, 4, 200); ax[2].plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi), "k-", lw=1.8, label="N(0,1)")
ax[2].set_title("Embedding projection vs normal"); ax[2].legend(frameon=False, fontsize=9)
ax[3].hist(vf, bins=40, alpha=.6, color="#B5642A", label="if FASTER")
ax[3].hist(vs, bins=40, alpha=.6, color="#2f5aa8", label="if SLOWER")
ax[3].set_xlabel("predicted next ego speed"); ax[3].set_title("Same state, different action\n(causality)"); ax[3].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig("lewm2_show.png", dpi=150); print("saved -> lewm2_show.png")
