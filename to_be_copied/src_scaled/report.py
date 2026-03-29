"""
report.py  (scaled)

Runs one evaluation episode and writes a full text report to
results_scaled/evaluation_report.txt

Report contains:
  - Final summary statistics
  - Per-UAV path day-by-day
  - Per-day action log
  - ASCII grid snapshots every 12 days

Usage:
    python report.py
    python report.py --weights-dir weights_scaled/ --out-dir results_scaled/
"""

import os
import sys
import argparse
import numpy as np
import torch

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(SCRIPT_DIR)
ROOT_DIR      = os.path.dirname(PROJECT_DIR)
SIM_LOG_PATH  = os.path.join(ROOT_DIR, 'simulation_scaled', 'simulation_log.csv')
GRID_CFG_PATH = os.path.join(ROOT_DIR, 'grid_scaled', 'grid_config.json')
DEFAULT_WDIR  = os.path.join(PROJECT_DIR, 'weights_scaled')
DEFAULT_ODIR  = os.path.join(PROJECT_DIR, 'results_scaled')

sys.path.insert(0, SCRIPT_DIR)
from uav_env import (UAVFieldEnv, GRID_ROWS, GRID_COLS, N_SECTORS,
                     N_UAVS, E_MAX, ACTIONS)
from networks import SectorAttentionActor

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACTION_NAMES = {0: 'STAY', 1: 'NORTH', 2: 'SOUTH', 3: 'WEST', 4: 'EAST'}


def make_ascii_grid(env, uav_pos_list):
    """Returns a multi-line ASCII string of the current grid state."""
    pos_set = {uav_pos_list[u]: u for u in range(N_UAVS)}
    lines   = ["     " + "".join(f"{c:3}" for c in range(GRID_COLS))]
    for r in range(GRID_ROWS):
        row = f"r{r:2}  "
        for c in range(GRID_COLS):
            sid = r * GRID_COLS + c
            if (r, c) in pos_set:
                cell = str(pos_set[(r, c)])
            else:
                cell = ["H", "I", "?"][env.uav_status[sid]]
            row += f"{cell:>3}"
        lines.append(row)
    return "\n".join(lines)


def run_and_report(actors, env, out_path):
    obs       = env.reset()
    log_lines = []
    path_log  = [[] for _ in range(N_UAVS)]   # list of (day, row, col) per UAV
    rewards_per_day = []

    for t in range(env.T):
        actions = []
        for u in range(N_UAVS):
            obs_t = torch.FloatTensor(obs[u]).to(DEVICE)
            with torch.no_grad():
                action, _ = actors[u].get_action(obs_t)
            actions.append(action)
            path_log[u].append((t, *env.uav_pos[u]))

        prev_unknown = set(np.where(env.uav_status == 2)[0])
        obs, rewards, done, info = env.step(actions)

        newly_found = prev_unknown - set(np.where(env.uav_status == 2)[0])
        rewards_per_day.append(sum(rewards))

        action_str = "  ".join(
            f"UAV{u}:{ACTION_NAMES[actions[u]]}→{info['uav_pos'][u]}"
            for u in range(N_UAVS)
        )
        log_lines.append(
            f"  Day {t+1:>3}  {action_str}"
            f"  new={len(newly_found)}  reward={sum(rewards):.2f}"
        )

        if done:
            break

    # Final positions
    for u in range(N_UAVS):
        path_log[u].append((env.t, *env.uav_pos[u]))

    # ── Compose report ────────────────────────────────────────────────────────
    n_vis = int((env.uav_status != 2).sum())
    n_inf = int((env.uav_status == 1).sum())
    n_tru = int((env.true_status == 1).sum())

    lines = []
    sep   = "=" * 70

    lines += [
        sep,
        "  EVALUATION REPORT — SCALED MAPPO (10×10 Grid, 4 UAVs)",
        sep,
        f"  Grid            : {GRID_ROWS} rows × {GRID_COLS} cols = {N_SECTORS} sectors",
        f"  UAVs            : {N_UAVS}",
        f"  Episode length  : {env.T} days",
        f"  Total reward    : {sum(rewards_per_day):.2f}",
        "",
        "  ── Detection Results ──",
        f"  Sectors visited       : {n_vis} / {N_SECTORS}"
        f"  ({n_vis/N_SECTORS*100:.1f}%)",
        f"  Infected found        : {n_inf}",
        f"  True infected (final) : {n_tru}",
        f"  Detection rate        : {n_inf / max(n_tru, 1) * 100:.1f}%",
        "",
        "  ── Energy Remaining ──",
    ]
    for u in range(N_UAVS):
        lines.append(f"  UAV {u}  : {env.energy[u]:.1f} / {E_MAX:.0f}")

    # Path summary
    lines += ["", sep, "  UAV PATHS (day, row, col)", sep]
    for u in range(N_UAVS):
        lines.append(f"\n  UAV {u}:")
        chunks = [path_log[u][i:i+10] for i in range(0, len(path_log[u]), 10)]
        for chunk in chunks:
            lines.append("    " + "   ".join(
                f"d{d:02d}→({r},{c})" for d, r, c in chunk
            ))

    # Per-day action log
    lines += ["", sep, "  PER-DAY ACTION LOG", sep]
    lines += log_lines

    # ASCII grid snapshots every 12 days
    lines += ["", sep, "  GRID SNAPSHOTS (every 12 days)", sep]

    obs2 = env.reset()
    snap_env_actions = []
    for t in range(env.T):
        actions = []
        for u in range(N_UAVS):
            obs_t = torch.FloatTensor(obs2[u]).to(DEVICE)
            with torch.no_grad():
                action, _ = actors[u].get_action(obs_t)
            actions.append(action)
        snap_env_actions.append(actions)
        obs2, _, done2, _ = env.step(actions)
        if t % 12 == 0 or done2:
            lines.append(f"\n  t={t+1}")
            lines.append(make_ascii_grid(env, list(env.uav_pos)))
            lines.append(f"  Risk (mean={env.w.mean():.3f}  max={env.w.max():.3f})")
        if done2:
            break

    # Sector ID map
    lines += ["", sep, "  SECTOR ID MAP", sep,
              "     " + "".join(f"{c:4}" for c in range(GRID_COLS))]
    for r in range(GRID_ROWS):
        row = f"r{r:2}  "
        for c in range(GRID_COLS):
            row += f"{r*GRID_COLS+c:4}"
        lines.append(row)

    report = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=str, default=DEFAULT_WDIR)
    parser.add_argument('--out-dir',     type=str, default=DEFAULT_ODIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'evaluation_report.txt')

    env    = UAVFieldEnv(SIM_LOG_PATH, GRID_CFG_PATH)
    actors = [SectorAttentionActor().to(DEVICE) for _ in range(N_UAVS)]
    for u in range(N_UAVS):
        path = os.path.join(args.weights_dir, f'actor{u}_final.pth')
        actors[u].load_state_dict(torch.load(path, map_location=DEVICE))
        actors[u].eval()

    print('Generating report...')
    report = run_and_report(actors, env, out_path)
    print(report)
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
