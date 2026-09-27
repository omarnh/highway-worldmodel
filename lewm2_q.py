"""
lewm2_q.py — Q = summed future surprise, one situation, a few action sequences.

Take an action -> get a representation; another action -> another representation;
a third -> a third. Surprise of each representation = 0.5*||z||^2 (-log p under
SIGReg's N(0,I)). Add the three -> Q for that action sequence.
"""
import numpy as np, torch
import gymnasium as gym, highway_env  # noqa
from lewm2 import LeWM2, N_HIST
from lewm2_data import CFG, SKIP, N_ACTIONS, onehot_block, P

ck = torch.load("lewm2_ckpt.pt", map_location="cpu", weights_only=False)
m = LeWM2(latent=ck["latent"]); m.load_state_dict(ck["model"]); m.eval()
mean, std = ck["obs_mean"], ck["obs_std"]
BLOCKS = torch.tensor(np.stack([onehot_block([a]*5) for a in range(N_ACTIONS)]), dtype=torch.float32)
NAMES = ["LEFT","IDLE","RIGHT","FASTER","SLOWER"]
def emb(o):
    with torch.no_grad(): return m.embed(torch.tensor((o.reshape(-1)-mean)/std,dtype=torch.float32).unsqueeze(0))[0]
def surprise(z): return 0.5*(z**2).sum().item()

# --- grab ONE surprising situation: the state with the highest 1-step surprise
rng=np.random.default_rng(0); env=gym.make("highway-fast-v0",config=CFG)
best=None
for ep in range(20):
    obs,_=env.reset(seed=ep); os_=[np.asarray(obs,np.float32).reshape(-1)]; ac=[]; done=False
    while not done:
        a=int(rng.choice(N_ACTIONS,p=P)); obs,_,te,tr,_=env.step(a); ac.append(a); os_.append(np.asarray(obs,np.float32).reshape(-1)); done=te or tr
    nf=len(ac)//SKIP
    if nf<4: continue
    kept=[os_[i*SKIP] for i in range(nf+1)]; bl=[onehot_block(ac[i*SKIP:(i+1)*SKIP]) for i in range(nf)]
    z=[emb(k) for k in kept]
    for h in range(N_HIST,len(kept)):
        with torch.no_grad():
            s=(m.predictor(torch.stack(z[h-N_HIST:h]).unsqueeze(0),torch.tensor(bl[h-1],dtype=torch.float32).unsqueeze(0))[0]-z[h]).norm().item()
        if best is None or s>best[0]: best=(s, torch.stack(z[h-N_HIST:h]))
env.close()
cur_s, hist = best
print(f"Chosen surprising situation: current 1-step surprise = {cur_s:.2f}\n")

# --- for a few action sequences, roll 3 steps and sum the surprises -> Q
def rollout_Q(hist, actions):
    z = hist.clone(); per=[]
    with torch.no_grad():
        for a in actions:
            zp = m.predictor(z.unsqueeze(0), BLOCKS[a].unsqueeze(0))[0]   # take action -> new representation
            per.append(surprise(zp))                                     # its surprise
            z = torch.stack([z[1], z[2], zp])                            # slide history
    return per, sum(per)

seqs = {"hold IDLE":[1,1,1], "brake SLOWER":[4,4,4], "accelerate FASTER":[3,3,3], "go RIGHT then hold":[2,1,1]}
print(f"{'action sequence':22s} {'surprise t+1':>12} {'t+2':>8} {'t+3':>8} {'Q(sum)':>9}")
for name, acts in seqs.items():
    per, Q = rollout_Q(hist, acts)
    print(f"{name:22s} {per[0]:12.2f} {per[1]:8.2f} {per[2]:8.2f} {Q:9.2f}")
print("\nLower Q = the 3-step future stays more 'expected' (less surprising) = better action.")
