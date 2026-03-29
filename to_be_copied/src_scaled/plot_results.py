"""
plot_results.py  (scaled)

Runs one evaluation episode and saves a comprehensive 8-panel
results figure to results_scaled/results_graph.png

Panels:
  1. Field coverage over time
  2. Infected found vs ground truth
  3. Battery drain per UAV
  4. Mean field risk weight
  5. Cumulative reward
  6. Per-day reward (green=positive, red=negative)
  7. UAV trajectory map
  8. Final grid heatmap (risk weights)

Usage:
    python plot_results.py
    python plot_results.py --weights-dir weights_scaled/ --out-dir results_scaled/
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(SCRIPT_DIR)
ROOT_DIR      = os.path.dirname(PROJECT_DIR)
SIM_LOG_PATH  = os.path.join(ROOT_DIR, 'simulation_scaled', 'simulation_log.csv')
GRID_CFG_PATH = os.path.join(ROOT_DIR, 'grid_scaled', 'grid_config.json')
DEFAULT_WDIR  = os.path.join(PROJECT_DIR, 'weights_scaled')
DEFAULT_ODIR  = os.path.join(PROJECT_DIR, 'results_scaled')

sys.path.insert(0, SCRIPT_DIR)
from uav_env import (UAVFieldEnv, GRID_ROWS, GRID_COLS, N_SECTORS,
                     N_UAVS, E_MAX)
from networks import SectorAttentionActor

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UAV_COLORS = ['dodgerblue', 'darkorange', 'mediumseagreen', 'crimson']


def run_episode(actors, env):
    """Run one full episode and collect per-step statistics."""
    obs      = env.reset()
    history  = []

    for t in range(env.T):
        actions = []
        for u in range(N_UAVS):
            obs_t = torch.FloatTensor(obs[u]).to(DEVICE)
            with torch.no_grad():
                action, _ = actors[u].get_action(obs_t)
            actions.append(action)

        obs, rewards, done, info = env.step(actions)
        history.append({
            't':           t + 1,
            'uav_pos':     list(env.uav_pos),
            'uav_status':  env.uav_status.copy(),
            'true_status': env.true_status.copy(),
            'risk_weights':env.w.copy(),
            'energy':      list(env.energy),
            'reward':      sum(rewards),
        })
        if done:
            break

    return history


def plot_results(history, env, out_path):
    timesteps    = [s['t'] for s in history]
    n_visited    = [(s['uav_status'] != 2).sum() for s in history]
    n_inf_found  = [(s['uav_status'] == 1).sum() for s in history]
    true_infected= [(s['true_status'] == 1).sum() for s in history]
    mean_risk    = [s['risk_weights'].mean() for s in history]
    rewards_day  = [s['reward'] for s in history]
    cum_reward   = np.cumsum(rewards_day)
    energy       = [[s['energy'][u] for s in history] for u in range(N_UAVS)]

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle('MAPPO Results (Scaled) — 10×10 Grid, 4 UAVs, 72 Days',
                 fontsize=15, fontweight='bold')

    # ── Panel 1: Coverage ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(timesteps, n_visited, color='steelblue', linewidth=2)
    ax.axhline(N_SECTORS, color='gray', linestyle='--', alpha=0.7,
               label=f'Total ({N_SECTORS})')
    ax.fill_between(timesteps, n_visited, alpha=0.15, color='steelblue')
    ax.set_title('Field Coverage Over Time')
    ax.set_xlabel('Day')
    ax.set_ylabel('Sectors Visited')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Detection ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(timesteps, true_infected, color='tomato', linestyle='--',
            linewidth=2, label='True Infected')
    ax.plot(timesteps, n_inf_found, color='darkred', linewidth=2,
            label='Found by UAVs')
    ax.fill_between(timesteps, n_inf_found, true_infected,
                    alpha=0.2, color='tomato', label='Detection Gap')
    ax.set_title('Infected Found vs Ground Truth')
    ax.set_xlabel('Day')
    ax.set_ylabel('Sectors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Battery ───────────────────────────────────────────────────────
    ax = axes[1, 0]
    for u in range(N_UAVS):
        ax.plot(timesteps, energy[u], color=UAV_COLORS[u],
                linewidth=1.8, label=f'UAV {u}')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Battery Remaining per UAV')
    ax.set_xlabel('Day')
    ax.set_ylabel('Energy Units')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Mean Risk ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(timesteps, mean_risk, color='darkorchid', linewidth=2)
    ax.fill_between(timesteps, mean_risk, alpha=0.15, color='darkorchid')
    ax.set_title('Mean Field Risk Weight Over Time')
    ax.set_xlabel('Day')
    ax.set_ylabel('Mean w[k]')
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Cumulative Reward ─────────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(timesteps, cum_reward, color='teal', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(timesteps, cum_reward, 0,
                    where=np.array(cum_reward) >= 0, alpha=0.2,
                    color='teal', label='Positive')
    ax.fill_between(timesteps, cum_reward, 0,
                    where=np.array(cum_reward) < 0, alpha=0.2,
                    color='red', label='Negative')
    ax.set_title('Cumulative Reward')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Per-Day Reward ────────────────────────────────────────────────
    ax = axes[2, 1]
    colors_r = ['mediumseagreen' if r >= 0 else 'tomato' for r in rewards_day]
    ax.bar(timesteps, rewards_day, color=colors_r, alpha=0.8, width=0.9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Per-Day Reward')
    ax.set_xlabel('Day')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.2, axis='y')

    # ── Panel 7: Trajectory Map ────────────────────────────────────────────────
    ax = axes[3, 0]
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.grid(True, alpha=0.25)
    ax.set_title('UAV Trajectories (○=start, □=end)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Shade sectors by final UAV knowledge
    final_status = history[-1]['uav_status']
    for sid in range(N_SECTORS):
        r, c = sid // GRID_COLS, sid % GRID_COLS
        color = ('#ffe0e0' if final_status[sid] == 1 else
                 '#e0ffe0' if final_status[sid] == 0 else '#f5f5f5')
        rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                              facecolor=color, edgecolor='#cccccc', linewidth=0.5)
        ax.add_patch(rect)

    # Mark seed infections
    init_true = history[0]['true_status'] if history else env.true_status
    for sid in range(N_SECTORS):
        if init_true[sid] == 1:
            r, c = sid // GRID_COLS, sid % GRID_COLS
            ax.text(c, r, '★', ha='center', va='center',
                    fontsize=6, color='darkred', alpha=0.6)

    for u in range(N_UAVS):
        pr = [s['uav_pos'][u][0] for s in history]
        pc = [s['uav_pos'][u][1] for s in history]
        ax.plot(pc, pr, color=UAV_COLORS[u], linewidth=1.2,
                alpha=0.55, label=f'UAV {u}')
        ax.plot(pc[0],  pr[0],  'o', color=UAV_COLORS[u], markersize=8, zorder=5)
        ax.plot(pc[-1], pr[-1], 's', color=UAV_COLORS[u], markersize=8, zorder=5)

    legend_patches = [Patch(facecolor='#ffe0e0', label='Infected (found)'),
                      Patch(facecolor='#e0ffe0', label='Healthy (found)'),
                      Patch(facecolor='#f5f5f5', label='Unknown')]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color=UAV_COLORS[u],
              linewidth=1.5, label=f'UAV {u}') for u in range(N_UAVS)],
              loc='lower right', fontsize=7, ncol=2)

    # ── Panel 8: Final Risk Heatmap ────────────────────────────────────────────
    ax = axes[3, 1]
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_title('Final Risk Weight Heatmap')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    final_risk = history[-1]['risk_weights']
    norm       = Normalize(vmin=0, vmax=1)
    cmap       = plt.cm.RdYlGn_r

    for sid in range(N_SECTORS):
        r, c  = sid // GRID_COLS, sid % GRID_COLS
        color = cmap(norm(final_risk[sid]))
        rect  = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                               facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

    # Mark final UAV positions
    for u, (r, c) in enumerate(history[-1]['uav_pos']):
        ax.text(c, r, str(u), ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor=UAV_COLORS[u],
                          edgecolor='none'))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Risk Weight', fraction=0.025, pad=0.03)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=str, default=DEFAULT_WDIR)
    parser.add_argument('--out-dir',     type=str, default=DEFAULT_ODIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env    = UAVFieldEnv(SIM_LOG_PATH, GRID_CFG_PATH)
    actors = [SectorAttentionActor().to(DEVICE) for _ in range(N_UAVS)]
    for u in range(N_UAVS):
        path = os.path.join(args.weights_dir, f'actor{u}_final.pth')
        actors[u].load_state_dict(torch.load(path, map_location=DEVICE))
        actors[u].eval()

    print('Running evaluation episode...')
    history = run_episode(actors, env)

    out_path = os.path.join(args.out_dir, 'results_graph.png')
    print('Generating results graph...')
    plot_results(history, env, out_path)


if __name__ == '__main__':
    main()
