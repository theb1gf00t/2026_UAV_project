"""
quick_test.py — 500-episode smoke test (~20 min on RTX 3050)

Verifies that the full training pipeline works end-to-end:
  env reset/step, rollout buffer, GAE, PPO update, reward shaping,
  treatment mechanic, checkpointing, and a short evaluation run.

Run from project/src_scaled/:
    python quick_test.py
"""

import os, sys, time, types
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
SIM_LOG   = os.path.join(ROOT, 'simulation_scaled', 'simulation_log.csv')
GRID_CFG  = os.path.join(ROOT, 'grid_scaled', 'grid_config.json')
DATASET   = os.path.join(ROOT, 'simulation_scaled', 'dataset.npy')
CKPT_DIR  = os.path.join(SRC_DIR, 'quick_test_weights')
os.makedirs(CKPT_DIR, exist_ok=True)
sys.path.insert(0, SRC_DIR)

from uav_env import (UAVFieldEnv, N_UAVS, N_SECTORS, PSI,
                     LAMBDA_ENG, ZETA, TAU_DIAG)
from networks import SectorAttentionActor, CriticNetwork

# ── Hyperparameters (same as notebook, just fewer episodes) ───────────────────
N_EPISODES        = 500
K_EPOCHS          = 6
MINI_BATCH_SIZE   = 24
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
EPS_CLIP          = 0.2
ENTROPY_COEFF     = 0.05
LR_ACTOR          = 3e-4
LR_CRITIC         = 3e-4
LR_MIN            = 1e-5
EXPLORE_BONUS     = 25.0
VISIT_BONUS       = 5.0
OVERHOVER_PENALTY = 5.0
LOG_INTERVAL      = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device  : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')

# ── Environment ────────────────────────────────────────────────────────────────
env = UAVFieldEnv(SIM_LOG, GRID_CFG, dataset_dir=DATASET)

def _new_compute_reward(self, u, energy_consumed):
    sid = self.pos_to_sid[self.uav_pos[u]]
    presence = PSI * self.w[sid] if self.uav_status[sid] == 2 else 0.0
    return presence - LAMBDA_ENG * energy_consumed - ZETA * self._compute_repulsion(u)

env._compute_reward = types.MethodType(_new_compute_reward, env)

# ── Models ─────────────────────────────────────────────────────────────────────
actors      = [SectorAttentionActor().to(DEVICE) for _ in range(N_UAVS)]
critic      = CriticNetwork().to(DEVICE)
actor_opts  = [torch.optim.Adam(a.parameters(), lr=LR_ACTOR) for a in actors]
critic_opt  = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
actor_scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=N_EPISODES, eta_min=LR_MIN)
                for o in actor_opts]
critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=N_EPISODES, eta_min=LR_MIN)

# ── Buffer ─────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def clear(self):
        self.obs=[[] for _ in range(N_UAVS)]; self.actions=[[] for _ in range(N_UAVS)]
        self.log_probs=[[] for _ in range(N_UAVS)]; self.rewards=[[] for _ in range(N_UAVS)]
        self.values=[]; self.dones=[]
    def __init__(self): self.clear()
    def store(self, o, a, lp, r, v, d):
        for u in range(N_UAVS):
            self.obs[u].append(o[u]); self.actions[u].append(a[u])
            self.log_probs[u].append(lp[u]); self.rewards[u].append(r[u])
        self.values.append(v); self.dones.append(d)
    def get_tensors(self):
        obs_t  = [torch.FloatTensor(np.array(self.obs[u])).to(DEVICE) for u in range(N_UAVS)]
        acts_t = [torch.LongTensor(self.actions[u]).to(DEVICE) for u in range(N_UAVS)]
        lps_t  = [torch.stack(self.log_probs[u]).to(DEVICE) for u in range(N_UAVS)]
        rews_t = [torch.FloatTensor(self.rewards[u]).to(DEVICE) for u in range(N_UAVS)]
        vals_t = torch.stack(self.values).view(-1).to(DEVICE)
        dones_t= torch.FloatTensor(self.dones).to(DEVICE)
        return obs_t, acts_t, lps_t, rews_t, vals_t, dones_t

def compute_gae(rews_list, values, dones, last_val):
    T = len(values)
    adv = torch.zeros(T).to(DEVICE)
    gae = 0.0
    mean_r = torch.stack(rews_list).mean(dim=0)
    vext   = torch.cat([values.view(-1), last_val.view(1)])
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = mean_r[t] + GAMMA * vext[t+1] * mask - vext[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        adv[t] = gae
    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.detach(), ret.detach()

def ppo_update(buffer):
    obs_t, acts_t, old_lps_t, rews_t, vals_t, dones_t = buffer.get_tensors()
    with torch.no_grad():
        lj = torch.cat([torch.FloatTensor(env._get_obs(u)).to(DEVICE)
                        for u in range(N_UAVS)]).unsqueeze(0)
        lv = critic(lj).squeeze()
    adv, ret = compute_gae(rews_t, vals_t, dones_t, lv)
    T = vals_t.shape[0]
    al, cl = [], []
    for _ in range(K_EPOCHS):
        perm = torch.randperm(T, device=DEVICE)
        for s in range(0, T, MINI_BATCH_SIZE):
            idx = perm[s:s+MINI_BATCH_SIZE]
            for u in range(N_UAVS):
                nlp, ent = actors[u].get_log_prob_entropy(obs_t[u][idx], acts_t[u][idx])
                ratio = torch.exp(nlp - old_lps_t[u][idx].detach())
                loss  = (-torch.min(ratio*adv[idx],
                          torch.clamp(ratio,1-EPS_CLIP,1+EPS_CLIP)*adv[idx]).mean()
                         - ENTROPY_COEFF * ent)
                actor_opts[u].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actors[u].parameters(), 0.5)
                actor_opts[u].step(); al.append(loss.item())
            jt = torch.cat([obs_t[u][idx] for u in range(N_UAVS)], dim=1)
            vp = critic(jt).squeeze()
            cl_loss = nn.MSELoss()(vp, ret[idx])
            critic_opt.zero_grad(); cl_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step(); cl.append(cl_loss.item())
    return np.mean(al), np.mean(cl)

# ── Training loop ──────────────────────────────────────────────────────────────
buf            = RolloutBuffer()
ep_rewards     = []
ep_discovered  = []
recent         = deque(maxlen=50)
t0             = time.time()

print(f'\nTraining for {N_EPISODES} episodes...\n')
print(f"{'Ep':>6}  {'Reward':>9}  {'Avg50':>9}  {'Found':>6}  {'ALoss':>7}  {'CLoss':>7}  {'ETA':>8}")
print('─' * 65)

for ep in range(1, N_EPISODES + 1):
    obs = env.reset(); buf.clear(); ep_r = 0.0

    for t in range(env.T):
        jt = torch.cat([torch.FloatTensor(obs[u]).to(DEVICE)
                        for u in range(N_UAVS)]).unsqueeze(0)
        with torch.no_grad(): v = critic(jt)
        acts, lps = [], []
        for u in range(N_UAVS):
            with torch.no_grad():
                a, lp = actors[u].get_action(torch.FloatTensor(obs[u]).to(DEVICE))
            acts.append(a); lps.append(lp)

        unk_before = set(np.where(env.uav_status == 2)[0])
        nobs, rewards, done, _ = env.step(acts)

        newly = unk_before - set(np.where(env.uav_status == 2)[0])
        if newly:
            for u in range(N_UAVS): rewards[u] += EXPLORE_BONUS * len(newly)
        for u in range(N_UAVS):
            sid = env.pos_to_sid[env.uav_pos[u]]
            if env.uav_status[sid] == 2:   rewards[u] += VISIT_BONUS
            elif env.dwell[u] > TAU_DIAG:  rewards[u] -= OVERHOVER_PENALTY

        buf.store(obs, acts, lps, rewards, v, float(done))
        ep_r += sum(rewards); obs = nobs
        if done: break

    for u in range(N_UAVS):
        r = np.array(buf.rewards[u], dtype=np.float32)
        buf.rewards[u] = list((r - r.mean()) / (r.std() + 1e-8))

    al, cl = ppo_update(buf)
    for s in actor_scheds: s.step()
    critic_sched.step()

    n_found = int((env.uav_status == 1).sum())
    ep_rewards.append(ep_r); ep_discovered.append(n_found); recent.append(ep_r)
    avg = np.mean(recent)

    if ep % LOG_INTERVAL == 0 or ep == 1:
        elapsed = time.time() - t0
        eta     = (elapsed / ep) * (N_EPISODES - ep)
        print(f'{ep:>6}  {ep_r:>9.1f}  {avg:>9.1f}  {n_found:>6}  '
              f'{al:>7.4f}  {cl:>7.4f}  {eta/60:>6.1f}min')

total = time.time() - t0
print(f'\nDone in {total/60:.1f} min')
print(f'Best reward  : {max(ep_rewards):.2f}')
print(f'Avg(last 50) : {np.mean(list(recent)):.2f}')
print(f'Best found   : {max(ep_discovered)}/{N_SECTORS}')

# ── Quick eval ─────────────────────────────────────────────────────────────────
print('\nRunning evaluation episode...')
eval_env = UAVFieldEnv(SIM_LOG, GRID_CFG, dataset_dir=DATASET)
e_obs    = eval_env.reset()
e_reward = 0.0
for a in actors: a.eval()

for t in range(eval_env.T):
    e_acts = []
    for u in range(N_UAVS):
        with torch.no_grad():
            a, _ = actors[u].get_action(torch.FloatTensor(e_obs[u]).to(DEVICE))
        e_acts.append(a)
    e_obs, r, done, _ = eval_env.step(e_acts)
    e_reward += sum(r)
    if done: break

n_vis = int((eval_env.uav_status != 2).sum())
n_inf = int((eval_env.uav_status == 1).sum())
n_tru = int((eval_env.true_status == 1).sum())
print(f'  Sectors visited : {n_vis}/{N_SECTORS}')
print(f'  Infected found  : {n_inf}  |  True infected: {n_tru}')
print(f'  Detection rate  : {n_inf/max(n_tru,1)*100:.1f}%')

# Save weights
for u in range(N_UAVS):
    torch.save(actors[u].state_dict(), os.path.join(CKPT_DIR, f'actor{u}_quicktest.pth'))
torch.save(critic.state_dict(), os.path.join(CKPT_DIR, 'critic_quicktest.pth'))
print(f'\nWeights saved to {CKPT_DIR}/')
