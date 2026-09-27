"""
jepa.py — the model pieces for a latent-space JEPA world model.

Three objects and one diagnostic, which is the whole point of this file:

  Encoder     f_theta : state  -> latent z            (online, trained by grad)
  Encoder     f_xi    : state  -> latent z (target)   (EMA copy, no grad)
  Predictor   g_phi   : (z_t, a_t) -> z_hat_{t+1}      (online, trained by grad)
  collapse_metrics()  : is the encoder cheating by mapping everything to a
                        constant? This is the failure mode to watch.

The learning signal (built in train.py) is:

    z_hat = g_phi(f_theta(s_t), a_t)          # predict next latent
    z_tgt = f_xi(s_{t+1}).detach()            # target latent, STOP-GRADIENT
    loss  = d(z_hat, z_tgt)                    # pull them together

Nothing pushes latents *apart*. So why doesn't the encoder collapse to a
constant (which would make the loss trivially zero)? Because of the
asymmetry: the target comes from an EMA encoder that lags behind, and only
the online side has the predictor. That lag + predictor is the BYOL trick
that makes a no-negatives objective stable. If it collapses anyway, flip on
the VICReg variance term in train.py — that's the honest backstop.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.GELU):
    """Small helper: build an MLP from a list of layer sizes."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """f: state -> latent z. Same class is used for online and target."""

    def __init__(self, obs_dim, latent_dim=32, hidden=128):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, latent_dim])

    def forward(self, s):
        return self.net(s)


class Predictor(nn.Module):
    """g: (z_t, a_t) -> predicted z_{t+1}. Action-conditioned."""

    def __init__(self, latent_dim=32, act_dim=1, hidden=128):
        super().__init__()
        self.net = mlp([latent_dim + act_dim, hidden, hidden, latent_dim])

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class JEPA(nn.Module):
    """
    Bundles online encoder + target encoder + predictor and owns the EMA
    update. Keep the target's parameters out of the optimizer — this module
    exposes .online_parameters() so train.py can't accidentally optimize the
    target.
    """

    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden=128, ema_tau=0.996):
        super().__init__()
        self.online_encoder = Encoder(obs_dim, latent_dim, hidden)
        self.predictor = Predictor(latent_dim, act_dim, hidden)

        # Target encoder starts as an exact copy of the online encoder, then
        # only ever moves via EMA. Freeze its grads.
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.ema_tau = ema_tau

    def online_parameters(self):
        """Exactly the params the optimizer should touch (NOT the target)."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    def forward(self, s_t, a_t, s_next):
        """
        Returns (z_hat, z_tgt) already prepared for the loss:
          z_hat = predictor(online_encode(s_t), a_t)
          z_tgt = target_encode(s_next), detached (stop-gradient)
        """
        z_t = self.online_encoder(s_t)
        z_hat = self.predictor(z_t, a_t)
        with torch.no_grad():
            z_tgt = self.target_encoder(s_next)
        return z_hat, z_tgt.detach()

    @torch.no_grad()
    def ema_update(self):
        """xi <- tau*xi + (1-tau)*theta. Call once per optimizer step."""
        tau = self.ema_tau
        for p_online, p_target in zip(self.online_encoder.parameters(),
                                      self.target_encoder.parameters()):
            p_target.mul_(tau).add_(p_online.data, alpha=1 - tau)


def byol_loss(z_hat, z_tgt):
    """
    Normalized L2 (== 2 - 2*cosine). Scale-invariant, which stops the model
    from 'winning' by shrinking the norm of z. This is the default objective.
    """
    z_hat = F.normalize(z_hat, dim=-1)
    z_tgt = F.normalize(z_tgt, dim=-1)
    return (2 - 2 * (z_hat * z_tgt).sum(dim=-1)).mean()


def vicreg_variance(z, eps=1e-4, target_std=1.0):
    """
    VICReg variance term: hinge that pushes each latent dim's std up toward
    target_std. Add this (small weight) only if you SEE collapse in the logs.
    It's the safety net, not the main mechanism.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


@torch.no_grad()
def collapse_metrics(z):
    """
    The diagnostic. Give it a batch of latents z (shape [B, D]) and get back
    numbers that tell you whether the encoder is still saying anything:

      latent_std      mean over dims of per-dim std across the batch.
                      -> 0 means collapse (every input maps to ~same point).
      active_dims     # of dims whose std > 0.01 (how many are 'alive').
      eff_rank        effective rank from the covariance eigenvalue spectrum
                      (entropy-based). Low eff_rank = representation is thin.
    """
    std = z.std(dim=0)
    latent_std = std.mean().item()
    active_dims = int((std > 1e-2).sum().item())

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = ev / ev.sum()
    eff_rank = float(torch.exp(-(p * p.log()).sum()).item())

    return {"latent_std": latent_std, "active_dims": active_dims, "eff_rank": eff_rank}
