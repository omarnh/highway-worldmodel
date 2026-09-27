"""
lewm2_data.py — collect frame-skip-5 sub-trajectories from highway for the
faithful LeWorldModel training (history N=3, sub-trajectory length 4).

Per env step we record (obs, action). We then keep every 5th frame (frame-skip 5)
and group the 5 env-actions between kept frames into one ACTION BLOCK, encoded as
5 concatenated one-hots (25-dim). We build windows of 4 consecutive kept frames
[f0,f1,f2,f3] with the 3 action blocks between them.

Data-collection policy is survival-biased (favor IDLE/SLOWER) so episodes last long
enough to yield length-4 windows under frame-skip 5. (Disclosed adaptation.)
"""
import numpy as np
import gymnasium as gym
import highway_env  # noqa

ENV_ID = "highway-fast-v0"
CFG = {"vehicles_count": 15, "simulation_frequency": 10, "policy_frequency": 2, "duration": 40}
N_ACTIONS = 5
SKIP = 5
WIN = 4                     # frames per window (N=3 history + 1 target)
# meta-actions: 0 LANE_LEFT,1 IDLE,2 LANE_RIGHT,3 FASTER,4 SLOWER
P = np.array([0.125, 0.40, 0.125, 0.10, 0.25])


def onehot_block(acts):        # acts: list of 5 ints -> 25-dim
    z = np.zeros((5, N_ACTIONS), np.float32)
    z[np.arange(5), acts] = 1.0
    return z.reshape(-1)


def collect(target_windows, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID, config=CFG)
    F, B = [], []              # windows of frames, and their (WIN-1) action blocks
    ep = seed
    while len(F) < target_windows:
        obs, _ = env.reset(seed=ep); ep += 1
        obs_seq = [np.asarray(obs, np.float32).reshape(-1)]
        act_seq = []
        done = False
        while not done:
            a = int(rng.choice(N_ACTIONS, p=P))
            obs, _, term, trunc, _ = env.step(a)
            act_seq.append(a)
            obs_seq.append(np.asarray(obs, np.float32).reshape(-1))
            done = term or trunc
        # keep every 5th frame; need full blocks of 5 actions between kept frames
        n_full = (len(act_seq)) // SKIP           # number of complete 5-action gaps
        if n_full < WIN - 1:
            continue
        kept = [obs_seq[i * SKIP] for i in range(n_full + 1)]         # n_full+1 frames
        blocks = [onehot_block(act_seq[i * SKIP:(i + 1) * SKIP]) for i in range(n_full)]
        for j in range(0, (n_full + 1) - WIN + 1):
            F.append(np.stack(kept[j:j + WIN]))                       # [WIN,25]
            B.append(np.stack(blocks[j:j + WIN - 1]))                 # [WIN-1,25]
    env.close()
    return np.array(F, np.float32), np.array(B, np.float32)


if __name__ == "__main__":
    F, B = collect(3000, seed=0)
    Fte, Bte = collect(800, seed=9999)
    flat = F.reshape(-1, 25)
    obs_mean = flat.mean(0); obs_std = flat.std(0) + 1e-6
    np.savez("lewm2_data.npz", F=F, B=B, Fte=Fte, Bte=Bte,
             obs_mean=obs_mean, obs_std=obs_std)
    print(f"train windows {len(F)}  test {len(Fte)}  frame shape {F.shape}  block shape {B.shape}")
