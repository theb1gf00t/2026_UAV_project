"""
train.py  (scaled)

MAPPO training for the 10×10 grid, 4-UAV, 72-day system.

Key upgrades over original:
  - Generalised for N_UAVS actors (no hardcoded UAV 0/1 references)
  - Mini-batch PPO: 72-step rollout split into MINI_BATCH_SIZE=24 chunks
  - Cosine annealing LR scheduler over 20 000 episodes
  - Attention-based actors (~345K params each) + large MLP critic (~593K)
  - Total: ~1.97M parameters

Usage:
    python train.py
    python train.py --episodes 20000 --save-dir weights_scaled/
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(SCRIPT_DIR)
ROOT_DIR      = os.path.dirname(PROJECT_DIR)
SIM_LOG_PATH  = os.path.join(ROOT_DIR, 'simulation_scaled', 'simulation_log.csv')
GRID_CFG_PATH = os.path.join(ROOT_DIR, 'grid_scaled', 'grid_config.json')
DATASET_DIR   = os.path.join(ROOT_DIR, 'simulation_scaled', 'dataset.npy')
DEFAULT_SAVE  = os.path.join(PROJECT_DIR, 'weights_scaled')

sys.path.insert(0, SCRIPT_DIR)
from uav_env import UAVFieldEnv, N_UAVS, N_SECTORS
from networks import SectorAttentionActor, CriticNetwork

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EPISODES      = 20_000
K_EPOCHS        = 15
MINI_BATCH_SIZE = 24       # 72 steps / 3 mini-batches
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
EPS_CLIP        = 0.2
ENTROPY_COEFF   = 0.05
LR_ACTOR        = 3e-4
LR_CRITIC       = 3e-4
LR_MIN          = 1e-5     # cosine annealing floor
SAVE_INTERVAL   = 1_000
LOG_INTERVAL    = 50
EXPLORE_BONUS   = 3.0      # lower than original — fires more often on 100 sectors

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Rollout Buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs       = [[] for _ in range(N_UAVS)]
        self.actions   = [[] for _ in range(N_UAVS)]
        self.log_probs = [[] for _ in range(N_UAVS)]
        self.rewards   = [[] for _ in range(N_UAVS)]
        self.values    = []
        self.dones     = []

    def store(self, obs_list, actions_list, log_probs_list,
              rewards_list, value, done):
        for u in range(N_UAVS):
            self.obs[u].append(obs_list[u])
            self.actions[u].append(actions_list[u])
            self.log_probs[u].append(log_probs_list[u])
            self.rewards[u].append(rewards_list[u])
        self.values.append(value)
        self.dones.append(done)

    def get_tensors(self):
        obs_t   = [torch.FloatTensor(np.array(self.obs[u])).to(DEVICE)
                   for u in range(N_UAVS)]
        acts_t  = [torch.LongTensor(self.actions[u]).to(DEVICE)
                   for u in range(N_UAVS)]
        lps_t   = [torch.stack(self.log_probs[u]).to(DEVICE)
                   for u in range(N_UAVS)]
        rews_t  = [torch.FloatTensor(self.rewards[u]).to(DEVICE)
                   for u in range(N_UAVS)]
        vals_t  = torch.stack(self.values).view(-1).to(DEVICE)
        dones_t = torch.FloatTensor(self.dones).to(DEVICE)
        return obs_t, acts_t, lps_t, rews_t, vals_t, dones_t


# ── GAE ────────────────────────────────────────────────────────────────────────
def compute_gae(rewards_list, values, dones, last_value):
    T_ep       = len(values)
    advantages = torch.zeros(T_ep).to(DEVICE)
    gae        = 0.0
    mean_rews  = torch.stack(rewards_list).mean(dim=0)
    values_ext = torch.cat([values.view(-1), last_value.view(1)])

    for t in reversed(range(T_ep)):
        mask          = 1.0 - dones[t]
        delta         = mean_rews[t] + GAMMA * values_ext[t + 1] * mask - values_ext[t]
        gae           = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages[t] = gae

    returns    = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.detach(), returns.detach()


# ── Mini-Batch PPO Update ──────────────────────────────────────────────────────
def ppo_update(env, actors, critic, actor_opts, critic_opt, buffer):
    obs_t, acts_t, old_lps_t, rews_t, vals_t, dones_t = buffer.get_tensors()

    # Compute last value for GAE bootstrap
    with torch.no_grad():
        last_joint = torch.cat(
            [torch.FloatTensor(env._get_obs(u)).to(DEVICE) for u in range(N_UAVS)]
        ).unsqueeze(0)
        last_value = critic(last_joint).squeeze()

    advantages, returns = compute_gae(rews_t, vals_t, dones_t, last_value)

    T_ep         = vals_t.shape[0]
    actor_losses  = []
    critic_losses = []

    for _ in range(K_EPOCHS):
        # Shuffle timesteps for mini-batching
        perm = torch.randperm(T_ep, device=DEVICE)

        for start in range(0, T_ep, MINI_BATCH_SIZE):
            mb_idx = perm[start: start + MINI_BATCH_SIZE]

            mb_adv     = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            # ── Actor updates ─────────────────────────────────────────────────
            for u in range(N_UAVS):
                mb_obs     = obs_t[u][mb_idx]
                mb_acts    = acts_t[u][mb_idx]
                mb_old_lps = old_lps_t[u][mb_idx].detach()

                new_lps, entropy = actors[u].get_log_prob_entropy(mb_obs, mb_acts)
                ratio            = torch.exp(new_lps - mb_old_lps)
                surr1            = ratio * mb_adv
                surr2            = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv
                actor_loss       = (-torch.min(surr1, surr2).mean()
                                    - ENTROPY_COEFF * entropy)

                actor_opts[u].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[u].parameters(), 0.5)
                actor_opts[u].step()
                actor_losses.append(actor_loss.item())

            # ── Critic update ─────────────────────────────────────────────────
            mb_joint     = torch.cat([obs_t[u][mb_idx] for u in range(N_UAVS)], dim=1)
            values_pred  = critic(mb_joint).squeeze()
            critic_loss  = nn.MSELoss()(values_pred, mb_returns)

            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()
            critic_losses.append(critic_loss.item())

    return np.mean(actor_losses), np.mean(critic_losses)


# ── Training Curves ────────────────────────────────────────────────────────────
def plot_training_curves(ep_rewards, ep_discovered, a_losses, c_losses, save_dir):
    def ma(data, w=50):
        return np.convolve(data, np.ones(w) / w, mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('MAPPO Training (Scaled) — 10×10 Grid, 4 UAVs', fontsize=13)

    axes[0, 0].plot(ep_rewards, alpha=0.2, color='steelblue')
    if len(ep_rewards) >= 50:
        axes[0, 0].plot(ma(ep_rewards), color='steelblue', label='MA(50)')
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep_discovered, alpha=0.2, color='tomato')
    if len(ep_discovered) >= 50:
        axes[0, 1].plot(ma(ep_discovered), color='tomato', label='MA(50)')
    axes[0, 1].set_title('Infected Sectors Discovered per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if len(a_losses) >= 50:
        axes[1, 0].plot(ma(a_losses), color='darkorange')
    axes[1, 0].set_title('Actor Loss (MA-50)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].grid(True, alpha=0.3)

    if len(c_losses) >= 50:
        axes[1, 1].plot(ma(c_losses), color='mediumseagreen')
    axes[1, 1].set_title('Critic Loss (MA-50)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {out}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',    type=int,  default=N_EPISODES)
    parser.add_argument('--save-dir',    type=str,  default=DEFAULT_SAVE)
    parser.add_argument('--dataset-dir', type=str,  default=DATASET_DIR,
                        help='Path to dataset.npy (generate_dataset.py output). '
                             'Pass empty string "" to disable dataset mode.')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print('=' * 65)
    print('  MAPPO (Scaled) — 10×10 Grid | 4 UAVs | 72 Days')
    print('=' * 65)
    print(f'  Device       : {DEVICE}')
    if torch.cuda.is_available():
        print(f'  GPU          : {torch.cuda.get_device_name(0)}')
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  VRAM         : {mem:.1f} GB')
    print(f'  Episodes     : {args.episodes}')
    print(f'  Save dir     : {args.save_dir}')
    print('=' * 65)

    # ── Init ───────────────────────────────────────────────────────────────────
    _dset  = args.dataset_dir if args.dataset_dir else None
    env    = UAVFieldEnv(SIM_LOG_PATH, GRID_CFG_PATH, dataset_dir=_dset)
    actors = [SectorAttentionActor().to(DEVICE) for _ in range(N_UAVS)]
    critic = CriticNetwork().to(DEVICE)

    actor_opts = [torch.optim.Adam(a.parameters(), lr=LR_ACTOR) for a in actors]
    critic_opt  = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # Cosine annealing LR decay over full training
    actor_scheds = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.episodes, eta_min=LR_MIN)
        for opt in actor_opts
    ]
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_opt, T_max=args.episodes, eta_min=LR_MIN
    )

    actor_params  = sum(sum(p.numel() for p in a.parameters()) for a in actors)
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f'  Actor params : {actor_params:,}  (×{N_UAVS} = {actor_params:,})')
    print(f'  Critic params: {critic_params:,}')
    print(f'  Total params : {actor_params + critic_params:,}')
    print(f'  Episode len  : {env.T} steps')
    print(f'  Mini-batch   : {MINI_BATCH_SIZE} steps')
    print(f'  K_EPOCHS     : {K_EPOCHS}')
    print('=' * 65)
    print()

    buffer          = RolloutBuffer()
    ep_rewards      = []
    ep_discovered   = []
    a_loss_log      = []
    c_loss_log      = []
    recent_rewards  = deque(maxlen=50)
    train_start     = time.time()

    pbar = tqdm(range(1, args.episodes + 1), desc='Training', unit='ep',
                dynamic_ncols=True, colour='green')

    for episode in pbar:
        obs       = env.reset()
        buffer.clear()
        ep_reward = 0.0

        for t in range(env.T):
            # Joint obs for critic
            joint_obs = torch.cat(
                [torch.FloatTensor(obs[u]).to(DEVICE) for u in range(N_UAVS)]
            ).unsqueeze(0)

            with torch.no_grad():
                value = critic(joint_obs)

            actions   = []
            log_probs = []
            for u in range(N_UAVS):
                obs_t = torch.FloatTensor(obs[u]).to(DEVICE)
                with torch.no_grad():
                    action, lp = actors[u].get_action(obs_t)
                actions.append(action)
                log_probs.append(lp)

            unknown_before = set(np.where(env.uav_status == 2)[0])
            next_obs, rewards, done, info = env.step(actions)

            # Exploration bonus for newly diagnosed sectors
            unknown_after = set(np.where(env.uav_status == 2)[0])
            newly_found   = unknown_before - unknown_after
            if newly_found:
                for u in range(N_UAVS):
                    rewards[u] += EXPLORE_BONUS * len(newly_found)

            buffer.store(obs, actions, log_probs, rewards, value, float(done))
            ep_reward += sum(rewards)
            obs        = next_obs
            if done:
                break

        # Reward normalisation (z-score per episode per UAV)
        for u in range(N_UAVS):
            r = np.array(buffer.rewards[u], dtype=np.float32)
            buffer.rewards[u] = list((r - r.mean()) / (r.std() + 1e-8))

        a_loss, c_loss = ppo_update(
            env, actors, critic, actor_opts, critic_opt, buffer
        )

        # Step LR schedulers
        for sched in actor_scheds:
            sched.step()
        critic_sched.step()

        n_found = int((env.uav_status == 1).sum())
        ep_rewards.append(ep_reward)
        ep_discovered.append(n_found)
        a_loss_log.append(a_loss)
        c_loss_log.append(c_loss)
        recent_rewards.append(ep_reward)

        avg = np.mean(recent_rewards)
        pbar.set_postfix(ordered_dict={
            'reward': f'{ep_reward:.1f}',
            'avg50':  f'{avg:.1f}',
            'found':  n_found,
            'a_loss': f'{a_loss:.4f}',
            'c_loss': f'{c_loss:.4f}',
        })

        if episode % LOG_INTERVAL == 0 or episode == 1:
            elapsed = time.time() - train_start
            eta_sec = (elapsed / episode) * (args.episodes - episode)
            lr_now  = actor_opts[0].param_groups[0]['lr']
            tqdm.write(
                f'  Ep {episode:>6}/{args.episodes}'
                f'  reward={ep_reward:>9.2f}'
                f'  avg50={avg:>9.2f}'
                f'  found={n_found:>3}/{N_SECTORS}'
                f'  a_loss={a_loss:.4f}'
                f'  c_loss={c_loss:.4f}'
                f'  lr={lr_now:.2e}'
                f'  ETA={eta_sec/60:.1f}min'
            )

        if episode % SAVE_INTERVAL == 0:
            for u in range(N_UAVS):
                torch.save(actors[u].state_dict(),
                           os.path.join(ckpt_dir, f'actor{u}_ep{episode}.pth'))
            torch.save(critic.state_dict(),
                       os.path.join(ckpt_dir, f'critic_ep{episode}.pth'))
            tqdm.write(f'  [checkpoint] episode {episode}')

    # ── Save final weights ─────────────────────────────────────────────────────
    print('\nTraining complete. Saving final weights...')
    for u in range(N_UAVS):
        path = os.path.join(args.save_dir, f'actor{u}_final.pth')
        torch.save(actors[u].state_dict(), path)
        print(f'  [saved] {path}')
    path = os.path.join(args.save_dir, 'critic_final.pth')
    torch.save(critic.state_dict(), path)
    print(f'  [saved] {path}')

    # ── Training curves ────────────────────────────────────────────────────────
    print('\nGenerating training curves...')
    plot_training_curves(ep_rewards, ep_discovered, a_loss_log, c_loss_log,
                         args.save_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    print()
    print('=' * 65)
    print('  TRAINING SUMMARY')
    print('=' * 65)
    print(f'  Total time          : {total_time/60:.1f} min')
    print(f'  Best reward         : {max(ep_rewards):.2f}')
    print(f'  Final avg(50)       : {np.mean(list(recent_rewards)):.2f}')
    print(f'  Best infected found : {max(ep_discovered)} / {N_SECTORS}')
    print(f'  Weights saved to    : {args.save_dir}')
    print('=' * 65)


if __name__ == '__main__':
    main()
