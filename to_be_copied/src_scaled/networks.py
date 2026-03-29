"""
networks.py  (scaled)

Scaled-up neural networks for MAPPO UAV crop monitoring.

SectorAttentionActor  (~345K params per actor)
  Sector tokens (risk weight + status) are embedded and processed
  through a 2-layer Transformer encoder. The attended representation
  is combined with own state and other-UAV positions via an MLP.

  Architecture:
    sector_embed  : Linear(2 → 128)
    pos_embed     : Embedding(N_SECTORS, 128)  — learned 2-D position encoding
    transformer   : 2× TransformerEncoderLayer(d=128, heads=4, ffn=256)
    global_embed  : Linear(9 → 128)
    mlp           : Linear(256→256)→ReLU→Linear(256→5)→Categorical

CriticNetwork  (~593K params)
  Large MLP on the joint observation of all UAVs concatenated.

  Architecture:
    Linear(JOINT_SIZE→512)→ReLU→Linear(512→256)→ReLU
    →Linear(256→128)→ReLU→Linear(128→1)

Total model size: ~1.97M parameters
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from uav_env import N_SECTORS, N_UAVS, OBS_SIZE, JOINT_SIZE, N_ACTIONS

# Observation slice indices (must match uav_env._get_obs)
_OWN_END    = 3
_RISK_END   = 3 + N_SECTORS          # 103
_STATUS_END = 3 + N_SECTORS * 2      # 203
# [203 : OBS_SIZE] = other-UAV positions  (6 values for 4 UAVs)

_GLOBAL_DIM = _OWN_END + (N_UAVS - 1) * 2   # 3 + 6 = 9
_D_MODEL    = 128
_N_HEADS    = 4
_N_LAYERS   = 2
_FFN_DIM    = 256


class SectorAttentionActor(nn.Module):
    """
    Attention-based actor for one UAV.

    Forward input  : obs tensor of shape (..., OBS_SIZE)
    Forward output : Categorical distribution over N_ACTIONS actions
    """

    def __init__(self):
        super().__init__()

        # Sector feature embedding: [risk_weight, status_norm] → d_model
        self.sector_embed = nn.Linear(2, _D_MODEL)

        # Learned positional embedding indexed by sector_id (0 … N_SECTORS-1)
        self.pos_embed = nn.Embedding(N_SECTORS, _D_MODEL)

        # Transformer encoder (batch_first=True → (batch, seq, d))
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = _D_MODEL,
            nhead           = _N_HEADS,
            dim_feedforward = _FFN_DIM,
            dropout         = 0.0,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=_N_LAYERS)

        # Global feature embedding: [own_pos+energy + other_uav_rel_pos] → d_model
        self.global_embed = nn.Linear(_GLOBAL_DIM, _D_MODEL)

        # Final policy head: concat(sector_context, global_emb) → logits
        self.mlp = nn.Sequential(
            nn.Linear(_D_MODEL * 2, _FFN_DIM),
            nn.ReLU(),
            nn.Linear(_FFN_DIM, N_ACTIONS),
        )

        # Sector ID buffer — registered so it moves with .to(device)
        self.register_buffer(
            "sector_ids",
            torch.arange(N_SECTORS, dtype=torch.long)
        )

    def forward(self, obs):
        """
        obs : (..., OBS_SIZE)
        returns Categorical distribution
        """
        own_feats    = obs[..., :_OWN_END]                     # (..., 3)
        risk_w       = obs[..., _OWN_END:_RISK_END]            # (..., 100)
        status_n     = obs[..., _RISK_END:_STATUS_END]         # (..., 100)
        other_uav    = obs[..., _STATUS_END:]                  # (..., 6)

        # Build sector tokens: (..., N_SECTORS, 2)
        sector_feats = torch.stack([risk_w, status_n], dim=-1)

        # Handle arbitrary leading batch dims by flattening to (B, N, 2)
        leading      = sector_feats.shape[:-2]
        B            = 1
        for d in leading:
            B *= d
        sector_feats = sector_feats.view(B, N_SECTORS, 2)

        # Embed + add positional encoding
        sector_emb   = self.sector_embed(sector_feats)                 # (B, 100, 128)
        pos_emb      = self.pos_embed(self.sector_ids)                 # (100, 128)
        sector_emb   = sector_emb + pos_emb.unsqueeze(0)              # (B, 100, 128)

        # Transformer
        sector_ctx   = self.transformer(sector_emb)                    # (B, 100, 128)
        sector_pool  = sector_ctx.mean(dim=1)                          # (B, 128)

        # Global features
        global_feats = torch.cat(
            [own_feats.view(B, _OWN_END),
             other_uav.view(B, (N_UAVS - 1) * 2)], dim=-1
        )                                                               # (B, 9)
        global_emb   = self.global_embed(global_feats)                 # (B, 128)

        # Combine and produce logits
        combined     = torch.cat([sector_pool, global_emb], dim=-1)    # (B, 256)
        logits       = self.mlp(combined)                              # (B, 5)

        # Restore leading dims
        logits       = logits.view(*leading, N_ACTIONS) if leading else logits.squeeze(0)
        return Categorical(logits=logits)

    def get_action(self, obs):
        """Sample one action. obs : (OBS_SIZE,) or (1, OBS_SIZE)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist   = self.forward(obs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_log_prob_entropy(self, obs, actions):
        """Compute log-probs and mean entropy for a batch. obs : (T, OBS_SIZE)."""
        dist    = self.forward(obs)
        log_p   = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_p, entropy


class CriticNetwork(nn.Module):
    """
    Centralised critic — takes the joint observation of all UAVs.

    Input  : (batch, JOINT_SIZE=836)
    Output : (batch, 1)  — state value V(s)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(JOINT_SIZE, 512), nn.ReLU(),
            nn.Linear(512,        256), nn.ReLU(),
            nn.Linear(256,        128), nn.ReLU(),
            nn.Linear(128,          1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)


# ── Convenience: param count ──────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    actor  = SectorAttentionActor()
    critic = CriticNetwork()

    actor_p  = count_params(actor)
    critic_p = count_params(critic)
    total_p  = actor_p * N_UAVS + critic_p

    print(f"SectorAttentionActor : {actor_p:>10,} params  (×{N_UAVS} UAVs = {actor_p*N_UAVS:,})")
    print(f"CriticNetwork        : {critic_p:>10,} params")
    print(f"Total                : {total_p:>10,} params")

    # Smoke test
    import os, sys
    obs_batch = torch.randn(30, OBS_SIZE)
    dist      = actor(obs_batch)
    print(f"\nActor output (batch=30): {dist.logits.shape}  ✓")

    joint_batch = torch.randn(30, JOINT_SIZE)
    val         = critic(joint_batch)
    print(f"Critic output (batch=30): {val.shape}  ✓")
