"""
lewm2.py — faithful LeWorldModel training on highway vectors.

Faithful elements (from arXiv 2603.19312, Alg.1 / Eq.1-2):
  - single encoder + PROJECTOR (1-layer MLP + BatchNorm); prediction & SIGReg
    operate on the PROJECTED embedding.
  - predictor takes a HISTORY of N=3 frame embeddings + an action block, and
    predicts the next embedding DIRECTLY (not residual).
  - loss = teacher-forced single-step MSE (Eq.1) + lambda * SIGReg (Eq.2),
    SIGReg = Epps-Pulley ECF test (quadrature) over M projections vs N(0,I),
    applied to ALL frame embeddings in the batch. lambda = 0.1.
  - NO stop-gradient, NO EMA. Everything trained jointly.
Adaptations (disclosed): MLP encoder/predictor on the 25-d Kinematics vector
instead of ViT-on-pixels+AdaLN; action block = 5 concatenated one-hots (25-d).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from jepa import mlp, collapse_metrics

OBS_DIM = 25
BLOCK_DIM = 25            # 5 actions x 5 one-hot
N_HIST = 3
LATENT = 64


class Encoder(nn.Module):
    def __init__(self, latent=LATENT, hidden=128):
        super().__init__()
        self.net = mlp([OBS_DIM, hidden, hidden, latent])
    def forward(self, o): return self.net(o)


class Projector(nn.Module):
    """1-layer MLP + BatchNorm -> embedding (prediction & SIGReg live here)."""
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, latent)
        self.bn = nn.BatchNorm1d(latent)
    def forward(self, z): return self.bn(self.fc(z))


class Predictor(nn.Module):
    """[z_{t-2}, z_{t-1}, z_t, action_block] -> z_{t+1}  (direct)."""
    def __init__(self, latent=LATENT, hidden=256):
        super().__init__()
        self.net = mlp([N_HIST * latent + BLOCK_DIM, hidden, hidden, latent])
    def forward(self, hist, block):        # hist: [B, N_HIST, D], block: [B, BLOCK_DIM]
        x = torch.cat([hist.reshape(hist.shape[0], -1), block], dim=-1)
        return self.net(x)


def sigreg(Z, M=256, n_nodes=32, eps=1e-8):
    """Epps-Pulley ECF normality test vs N(0,1), quadrature over t in [0.2,4],
    averaged over M random unit projections. Z: [N, D]."""
    D = Z.shape[1]
    V = torch.randn(D, M, device=Z.device); V = V / (V.norm(dim=0, keepdim=True) + eps)
    H = Z @ V                                        # [N, M]
    t = torch.linspace(0.2, 4.0, n_nodes, device=Z.device)   # [T]
    th = H.unsqueeze(-1) * t                         # [N, M, T]
    re = torch.cos(th).mean(0)                       # [M, T]  Re phi_N
    im = torch.sin(th).mean(0)                       # [M, T]  Im phi_N
    phi0 = torch.exp(-0.5 * t ** 2)                  # [T]  standard-normal CF
    w = torch.exp(-0.5 * t ** 2)                     # [T]  Gaussian weight
    stat = (((re - phi0) ** 2 + im ** 2) * w).sum(-1)   # [M]
    return stat.mean()


class LeWM2(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.encoder = Encoder(latent)
        self.projector = Projector(latent)
        self.predictor = Predictor(latent)

    def embed(self, o):                              # o: [B, 25] -> [B, D]
        return self.projector(self.encoder(o))

    def loss(self, frames, blocks, lam=0.1):
        # frames: [B, 4, 25], blocks: [B, 3, 25]
        B, W, _ = frames.shape
        z = self.embed(frames.reshape(B * W, -1)).reshape(B, W, -1)   # [B,4,D]
        hist = z[:, :N_HIST]                          # [B,3,D]  (f0,f1,f2)
        a_last = blocks[:, N_HIST - 1]                # block f2->f3
        z_hat = self.predictor(hist, a_last)          # predict f3
        z_tgt = z[:, N_HIST]                          # true f3 (teacher-forced, NO detach)
        l_pred = F.mse_loss(z_hat, z_tgt)
        l_sig = sigreg(z.reshape(B * W, -1))          # all frame embeddings
        return l_pred + lam * l_sig, l_pred.item(), l_sig.item()
