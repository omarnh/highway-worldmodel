"""
lewm2_surprise.py — the surprise signal.

Surprise at each step = distance between the world model's 1-step prediction
(from the REAL current state + action) and the REAL next embedding:
    surprise_t = || predict(hist_t, action_t) - embed(o_{t+1}) ||

Normal driving -> the model predicts well -> low surprise.
Unexpected event -> the real next state jumps somewhere the model didn't expect
-> surprise spikes. We show this by injecting a sudden hazard (a car cuts in
close + ego forced to a sudden stop) at one step and watching the signal.
"""
import numpy as np, torch
import gymnasium as gym, highway_env  # noqa
from lewm2 import LeWM2, N_HIST
from lewm2_data import CFG, SKIP, N_ACTIONS, onehot_block, P

ck = torch.load("lewm2_ckpt.pt", map_location="cpu", weights_only=False)
m = LeWM2(latent=ck["latent"]); m.load_state_dict(ck["model"]); m.eval()
mean, std = ck["obs_mean"], ck["obs_std"]
def emb(o):
    with torch.no_grad(): return m.embed(torch.tensor((o.reshape(-1)-mean)/std, dtype=torch.float32).unsqueeze(0))[0]

def make_anomaly(obs):
    """Sudden hazard: a car cuts in very close ahead + ego abruptly slows."""
    o = obs.reshape(5,5).copy()
    o[0,3] = max(0.0, o[0,3]*0.1)                 # ego speed suddenly drops
    o[1] = [1.0, 0.03, 0.0, -0.3, 0.0]            # vehicle 1 appears right ahead, approaching
    return o.reshape(-1)

def collect_seq(nseq, L, seed):
    rng=np.random.default_rng(seed); env=gym.make("highway-fast-v0",config=CFG); seqs=[]; ep=seed
    while len(seqs)<nseq:
        obs,_=env.reset(seed=ep); ep+=1; os_=[np.asarray(obs,np.float32).reshape(-1)]; ac=[]; done=False
        while not done:
            a=int(rng.choice(N_ACTIONS,p=P)); obs,_,te,tr,_=env.step(a); ac.append(a)
            os_.append(np.asarray(obs,np.float32).reshape(-1)); done=te or tr
        nf=len(ac)//SKIP
        if nf<L: continue
        kept=np.stack([os_[i*SKIP] for i in range(L)]); bl=np.stack([onehot_block(ac[i*SKIP:(i+1)*SKIP]) for i in range(L-1)])
        seqs.append((kept,bl))
    env.close(); return seqs

L=9; seqs=collect_seq(200,L,seed=7)
def surprise(hist, block, znext):
    with torch.no_grad(): return (m.predictor(hist.unsqueeze(0), torch.tensor(block,dtype=torch.float32).unsqueeze(0))[0]-znext).norm().item()

# distributions: normal vs anomalous transitions
normal_s=[]; anom_s=[]
for kept,bl in seqs:
    z=torch.stack([emb(kept[i]) for i in range(L)])
    for h in range(N_HIST, L):
        hist=z[h-N_HIST:h]
        normal_s.append(surprise(hist, bl[h-1], z[h]))
        anom_s.append(surprise(hist, bl[h-1], emb(make_anomaly(kept[h]))))
normal_s=np.array(normal_s); anom_s=np.array(anom_s)
print(f"normal-step surprise:  mean {normal_s.mean():.3f}  (90th pct {np.percentile(normal_s,90):.3f})")
print(f"anomaly-step surprise: mean {anom_s.mean():.3f}")
print(f"anomaly is {anom_s.mean()/normal_s.mean():.1f}x the normal surprise")

# one representative episode trace with an anomaly injected mid-episode
kept,bl=seqs[0]; z=torch.stack([emb(kept[i]) for i in range(L)])
trace=[]; inj=L//2
for h in range(N_HIST,L):
    hist=z[h-N_HIST:h]
    nxt = emb(make_anomaly(kept[h])) if h==inj else z[h]
    trace.append(surprise(hist, bl[h-1], nxt))
base=[surprise(z[h-N_HIST:h], bl[h-1], z[h]) for h in range(N_HIST,L)]

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs=list(range(N_HIST,L))
fig,ax=plt.subplots(1,2,figsize=(12.5,4.4))
ax[0].plot(xs,base,"-o",color="#2E6F4E",label="normal driving")
ax[0].plot(xs,trace,"-o",color="#B5642A",label="unexpected event injected")
ax[0].axvline(inj,color="#B5642A",ls="--",lw=1); ax[0].set_xlabel("timestep (block)"); ax[0].set_ylabel("surprise")
ax[0].set_title("Surprise spikes on the unexpected event"); ax[0].legend(frameon=False,fontsize=9); ax[0].grid(alpha=.15)
ax[1].hist(normal_s,bins=40,alpha=.7,color="#2E6F4E",density=True,label="normal steps")
ax[1].hist(anom_s,bins=40,alpha=.6,color="#B5642A",density=True,label="anomaly steps")
ax[1].set_xlabel("surprise"); ax[1].set_title("Normal vs unexpected transitions"); ax[1].legend(frameon=False,fontsize=9)
plt.tight_layout(); plt.savefig("lewm2_surprise.png",dpi=150); print("saved -> lewm2_surprise.png")
