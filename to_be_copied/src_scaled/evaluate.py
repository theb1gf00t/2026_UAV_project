"""
evaluate.py  (scaled)

Load trained MAPPO weights and run one evaluation episode on the
10×10 grid with 4 UAVs.

Uses stochastic sampling (not greedy argmax) — see PROJECT_CONTEXT Bug #4.

Usage:
    python evaluate.py
    python evaluate.py --weights-dir weights_scaled/
    python evaluate.py --weights-dir weights_scaled/ --out-dir results_scaled/
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


# ── Snapshot ───────────────────────────────────────────────────────────────────
def snapshot(env, t, rewards):
    return {
        't':           t,
        'uav_pos':     list(env.uav_pos),
        'uav_status':  env.uav_status.copy(),
        'true_status': env.true_status.copy(),
        'risk_weights':env.w.copy(),
        'energy':      list(env.energy),
        'rewards':     list(rewards),
    }


# ── Grid frame ─────────────────────────────────────────────────────────────────
def plot_grid_frame(snap, out_dir):
    t            = snap['t']
    uav_pos      = snap['uav_pos']
    uav_status   = snap['uav_status']
    true_status  = snap['true_status']
    risk_weights = snap['risk_weights']
    energy       = snap['energy']

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f'Day {t:02d} — UAV Field Monitoring (10×10)', fontsize=13)

    ax = axes[0]
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('H=Healthy  I=Infected  ?=Unknown  (colour=risk weight)')

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.RdYlGn_r
    STATUS_SYM = {0: 'H', 1: 'I', 2: '?'}

    for sid in range(N_SECTORS):
        r, c  = sid // GRID_COLS, sid % GRID_COLS
        color = cmap(norm(risk_weights[sid]))
        rect  = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                               facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(c, r, STATUS_SYM[uav_status[sid]],
                ha='center', va='center', fontsize=7, fontweight='bold')
        if true_status[sid] == 1:
            ax.text(c + 0.38, r - 0.38, '★',
                    ha='center', va='center', fontsize=6, color='darkred')

    for u, (r, c) in enumerate(uav_pos):
        circle = plt.Circle((c, r), 0.28, color=UAV_COLORS[u],
                             zorder=5, alpha=0.85)
        ax.add_patch(circle)
        ax.text(c, r, str(u), ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Risk Weight', fraction=0.025, pad=0.03)

    ax2 = axes[1]
    ax2.axis('off')
    lines = [f'Day          : {t}', '']
    for u in range(N_UAVS):
        lines.append(f'UAV {u} pos   : {uav_pos[u]}')
        lines.append(f'UAV {u} energy: {energy[u]:.1f}/{E_MAX:.0f}')
        lines.append('')
    lines += [
        '── UAV Knowledge ──',
        f'Healthy  : {int((uav_status == 0).sum())}',
        f'Infected : {int((uav_status == 1).sum())}',
        f'Unknown  : {int((uav_status == 2).sum())}',
        '',
        '── Ground Truth ──',
        f'Infected : {int((true_status == 1).sum())}',
        '★ = infected sector',
    ]
    for i, line in enumerate(lines):
        ax2.text(0.02, 0.98 - i * 0.048, line,
                 transform=ax2.transAxes, fontsize=9,
                 fontfamily='monospace', va='top')

    plt.tight_layout()
    path = os.path.join(out_dir, 'frames', f'grid_t{t:02d}.png')
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()


# ── Trajectory plot ────────────────────────────────────────────────────────────
def plot_trajectories(history, out_dir):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.grid(True, alpha=0.3)
    ax.set_title('UAV Trajectories — 10×10 Grid', fontsize=13)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    for sid in range(N_SECTORS):
        r, c = sid // GRID_COLS, sid % GRID_COLS
        rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                              facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.6)
        ax.add_patch(rect)

    # Mark initial infections
    init_true = history[0]['true_status']
    for sid in range(N_SECTORS):
        if init_true[sid] == 1:
            r, c = sid // GRID_COLS, sid % GRID_COLS
            ax.text(c, r, '★', ha='center', va='center',
                    fontsize=8, color='darkred', alpha=0.7)

    for u in range(N_UAVS):
        pr = [s['uav_pos'][u][0] for s in history]
        pc = [s['uav_pos'][u][1] for s in history]
        ax.plot(pc, pr, color=UAV_COLORS[u], linewidth=1.5,
                alpha=0.6, label=f'UAV {u}')
        ax.plot(pc[0],  pr[0],  'o', color=UAV_COLORS[u], markersize=9, zorder=5)
        ax.plot(pc[-1], pr[-1], 's', color=UAV_COLORS[u], markersize=9, zorder=5)

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, 'uav_trajectories.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {path}')


# ── Summary stats ──────────────────────────────────────────────────────────────
def plot_summary_stats(history, out_dir):
    timesteps     = [s['t'] for s in history]
    n_visited     = [(s['uav_status'] != 2).sum() for s in history]
    n_inf_found   = [(s['uav_status'] == 1).sum() for s in history]
    true_infected = [(s['true_status'] == 1).sum() for s in history]
    mean_risk     = [s['risk_weights'].mean() for s in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Evaluation — 10×10 Grid, 4 UAVs', fontsize=13)

    axes[0, 0].plot(timesteps, n_visited, color='steelblue', linewidth=2)
    axes[0, 0].axhline(N_SECTORS, color='gray', linestyle='--', label='Total')
    axes[0, 0].set_title('Sectors Visited Over Time')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Sectors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(timesteps, true_infected, color='tomato',
                    linestyle='--', linewidth=2, label='True infected')
    axes[0, 1].plot(timesteps, n_inf_found, color='darkred',
                    linewidth=2, label='Found by UAVs')
    axes[0, 1].set_title('Infected Found vs Actual')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for u in range(N_UAVS):
        energy_u = [s['energy'][u] for s in history]
        axes[1, 0].plot(timesteps, energy_u, color=UAV_COLORS[u],
                        linewidth=1.5, label=f'UAV {u}')
    axes[1, 0].set_title('Energy Remaining per UAV')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Energy units')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(timesteps, mean_risk, color='darkorchid', linewidth=2)
    axes[1, 1].set_title('Mean Risk Weight Across Field')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Mean w[k]')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'evaluation_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {path}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=str, default=DEFAULT_WDIR)
    parser.add_argument('--out-dir',     type=str, default=DEFAULT_ODIR)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, 'frames'), exist_ok=True)

    print('=' * 60)
    print('  MAPPO Evaluation (Scaled) — 10×10 Grid, 4 UAVs')
    print('=' * 60)
    print(f'  Device       : {DEVICE}')
    if torch.cuda.is_available():
        print(f'  GPU          : {torch.cuda.get_device_name(0)}')
    print(f'  Weights dir  : {args.weights_dir}')
    print(f'  Output dir   : {args.out_dir}')
    print('=' * 60)

    env    = UAVFieldEnv(SIM_LOG_PATH, GRID_CFG_PATH)
    actors = [SectorAttentionActor().to(DEVICE) for _ in range(N_UAVS)]
    for u in range(N_UAVS):
        path = os.path.join(args.weights_dir, f'actor{u}_final.pth')
        actors[u].load_state_dict(torch.load(path, map_location=DEVICE))
        actors[u].eval()
        print(f'  Loaded: {path}')

    print('\nRunning evaluation episode...')
    obs          = env.reset()
    total_reward = 0.0
    history      = [snapshot(env, 0, [0.0] * N_UAVS)]

    for t in range(env.T):
        actions = []
        for u in range(N_UAVS):
            obs_t = torch.FloatTensor(obs[u]).to(DEVICE)
            with torch.no_grad():
                action, _ = actors[u].get_action(obs_t)
            actions.append(action)

        obs, rewards, done, info = env.step(actions)
        total_reward += sum(rewards)
        history.append(snapshot(env, t + 1, rewards))

        if (t + 1) % 12 == 0 or done:
            s     = history[-1]
            n_vis = int((s['uav_status'] != 2).sum())
            n_inf = int((s['uav_status'] == 1).sum())
            pos_str = '  '.join(f'UAV{u}@{s["uav_pos"][u]}' for u in range(N_UAVS))
            print(f'  Day {t+1:>3}  {pos_str}'
                  f'  visited={n_vis}/{N_SECTORS}  found={n_inf}'
                  f'  reward={sum(rewards):.2f}')
        if done:
            break

    print('\nGenerating plots...')
    for snap in history:
        if snap['t'] % 12 == 0:
            plot_grid_frame(snap, args.out_dir)
    n_frames = len([s for s in history if s['t'] % 12 == 0])
    print(f'  [saved] {args.out_dir}/frames/  ({n_frames} frames)')

    plot_trajectories(history, args.out_dir)
    plot_summary_stats(history, args.out_dir)

    final = history[-1]
    n_vis = int((final['uav_status'] != 2).sum())
    n_inf = int((final['uav_status'] == 1).sum())
    n_tru = int((final['true_status'] == 1).sum())

    print()
    print('=' * 60)
    print('  EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Total days            : {final["t"]}')
    print(f'  Total reward          : {total_reward:.2f}')
    print(f'  Sectors visited       : {n_vis} / {N_SECTORS}')
    print(f'  Infected found        : {n_inf}')
    print(f'  True infected (final) : {n_tru}')
    print(f'  Detection rate        : {n_inf / max(n_tru, 1) * 100:.1f}%')
    for u in range(N_UAVS):
        print(f'  UAV {u} energy left   : {final["energy"][u]:.1f} / {E_MAX:.0f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
