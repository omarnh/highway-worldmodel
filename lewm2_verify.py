import numpy as np, torch, torch.nn.functional as F
import gymnasium as gym, highway_env  # noqa
from lewm2 import LeWM2, N_HIST, sigreg
from jepa import collapse_metrics
from lewm2_data import CFG, SKIP, onehot_block, P, N_ACTIONS

ck = torch.load("lewm2_ckpt.pt", map_location="cpu", weights_only=False)
m = LeWM2(latent=ck["latent"]); m.load_state_dict(ck["model"]); m.eval()
mean, std = ck["obs_mean"], ck["obs_std"]
def norm(x): return (x - mean) / std

d = np.load("lewm2_data.npz"); F4, B3 = d["Fte"], d["Bte"]
Fn = torch.tensor(norm(F4), dtype=torch.float32); Bt = torch.tensor(B3, dtype=torch.float32)

with torch.no_grad():
    B, W, _ = Fn.shape
    z = m.embed(Fn.reshape(B * W, -1)).reshape(B, W, -1)
    z_hat = m.predictor(z[:, :N_HIST], Bt[:, N_HIST - 1])
    z_tgt = z[:, N_HIST]; z_prev = z[:, N_HIST - 1]
    z_shuf = m.predictor(z[:, :N_HIST], Bt[torch.randperm(B), N_HIST - 1])

def ce(a, b): return (2 - 2 * (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)).mean().item()
Zn = z.reshape(B * W, -1).numpy(); cm = collapse_metrics(torch.tensor(Zn))
cov = np.cov(Zn, rowvar=False); dvar = np.diag(cov); off = np.sqrt((cov[~np.eye(len(cov),dtype=bool)]**2).mean())
print("=== isotropy / collapse ===")
print(f"  latent_std {cm['latent_std']:.3f}  |mean| {np.abs(Zn.mean(0)).mean():.3f}  var[min/mean/max] {dvar.min():.2f}/{dvar.mean():.2f}/{dvar.max():.2f}  off-diag {off:.3f}  eff_rank {cm['eff_rank']:.1f}")
print("\n=== 1-step prediction (cosine, lower=better) ===")
print(f"  world model {ce(z_hat,z_tgt):.4f}   no-op {ce(z_prev,z_tgt):.4f}   shuffled-action {ce(z_shuf,z_tgt):.4f}")

# action counterfactual: FASTER block vs SLOWER block -> decoded ego speed
def block_of(a): return torch.tensor(np.tile(onehot_block([a]*5),(B,1)),dtype=torch.float32)
def r2(p,t): return 1-((t-p)**2).sum()/(((t-t.mean(0))**2).sum()+1e-12)
def lf(X,y): Xb=np.concatenate([X,np.ones((len(X),1))],1); w,*_=np.linalg.lstsq(Xb,y,rcond=None); return w
def lp(X,w): return np.concatenate([X,np.ones((len(X),1))],1)@w
# probe on frame embeddings (readout)
Ze = z.reshape(B*W,-1).numpy(); Yobs = F4.reshape(B*W,25); k=len(Ze)//2
print("\n=== probe readout R^2 (embedding -> scene quantity) ===")
for nm,col in {"ego_vx":3,"ego_y(lane)":2,"lead_dx(gap)":6,"lead_vx":8}.items():
    y=Yobs[:,col]; w=lf(Ze[:k],y[:k]); print(f"  {nm:14s} {r2(lp(Ze[k:],w),y[k:]):.3f}")

# ---- multi-step open-loop rollout vs persistence ----
def collect_seq(nseq, L, seed):
    rng=np.random.default_rng(seed); env=gym.make("highway-fast-v0",config=CFG); seqs=[]
    ep=seed
    while len(seqs)<nseq:
        obs,_=env.reset(seed=ep); ep+=1; os_=[np.asarray(obs,np.float32).reshape(-1)]; ac=[]; done=False
        while not done:
            a=int(rng.choice(N_ACTIONS,p=P)); obs,_,te,tr,_=env.step(a); ac.append(a); os_.append(np.asarray(obs,np.float32).reshape(-1)); done=te or tr
        nf=len(ac)//SKIP
        if nf<L: continue
        kept=[os_[i*SKIP] for i in range(nf+1)]; bl=[onehot_block(ac[i*SKIP:(i+1)*SKIP]) for i in range(nf)]
        seqs.append((np.stack(kept[:L]), np.stack(bl[:L-1])))
    env.close(); return seqs

L=9; seqs=collect_seq(150, L, seed=123)
# fit a readout on the test-window embeddings to decode ego_vx & gap
wq={c:lf(Ze, Yobs[:,c]) for c in [3,6]}
Hh=range(N_HIST, L)   # horizons we can predict (from frame idx N_HIST .. L-1)
roll_r2={3:[],6:[]}; pers_r2={3:[],6:[]}
with torch.no_grad():
    for c in [3,6]:
        for h in Hh:
            preds=[]; trues=[]; perss=[]
            for kept,bl in seqs:
                kf=torch.tensor(norm(kept),dtype=torch.float32)
                emb=m.embed(kf)                      # [L,D] real embeddings
                hist=emb[:N_HIST].clone()            # start history f0,f1,f2
                zt=None
                for step in range(N_HIST, h+1):
                    a=torch.tensor(bl[step-1],dtype=torch.float32).unsqueeze(0)
                    zt=m.predictor(hist.unsqueeze(0), a)[0]     # predict frame `step`
                    hist=torch.stack([hist[1],hist[2],zt])      # open-loop: feed prediction
                preds.append(lp(zt.numpy()[None],wq[c])[0]); trues.append(kept[h,c]); perss.append(kept[N_HIST-1,c])
            roll_r2[c].append(r2(np.array(preds),np.array(trues)))
            pers_r2[c].append(r2(np.array(perss),np.array(trues)))
print("\n=== multi-step open-loop rollout (R^2 vs true; persistence in parens) ===")
for c,nm in [(3,"ego_vx"),(6,"gap")]:
    print(f"  {nm}: " + "  ".join(f"h{h}={roll_r2[c][i]:.2f}({pers_r2[c][i]:.2f})" for i,h in enumerate(Hh)))

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15.5,4.4))
ax[0].bar(["world model","no-op","shuffled"],[ce(z_hat,z_tgt),ce(z_prev,z_tgt),ce(z_shuf,z_tgt)],color=["#2E6F4E","#7C8A96","#B5642A"])
ax[0].set_title("1-step next-embedding error\n(lower=better)"); ax[0].grid(alpha=.15,axis="y")
hs=list(Hh); steps=[h-(N_HIST-1) for h in hs]
ax[1].plot(steps,roll_r2[3],"-o",color="#2E6F4E",label="world model"); ax[1].plot(steps,pers_r2[3],"-o",color="#7C8A96",label="persistence")
ax[1].set_title("Multi-step forecast: ego speed"); ax[1].set_xlabel("steps ahead (x5 frames)"); ax[1].set_ylabel("R²"); ax[1].legend(frameon=False,fontsize=9); ax[1].grid(alpha=.15); ax[1].set_ylim(-1,1.05)
ax[2].plot(steps,roll_r2[6],"-o",color="#2E6F4E",label="world model"); ax[2].plot(steps,pers_r2[6],"-o",color="#7C8A96",label="persistence")
ax[2].set_title("Multi-step forecast: gap to lead"); ax[2].set_xlabel("steps ahead (x5 frames)"); ax[2].set_ylabel("R²"); ax[2].legend(frameon=False,fontsize=9); ax[2].grid(alpha=.15); ax[2].set_ylim(-1,1.05)
plt.tight_layout(); plt.savefig("lewm2_results.png",dpi=150); print("\nsaved -> lewm2_results.png")
