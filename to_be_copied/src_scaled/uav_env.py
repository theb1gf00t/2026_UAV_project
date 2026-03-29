"""
uav_env.py  (scaled)

Gym-style environment for the scaled-up Multi-UAV Active Sensing system.
10×10 grid, 4 UAVs, 72-day episodes.

Implements the same core equations as the original:
  Eq. 2  → Detection history H[k,t]
  Eq. 4  → Risk potential Omega[k,t]
  Eq. 5  → Dynamic risk weight w[k,t]
  Eq. 6  → Global risk density field
  Eq. 9  → Inter-UAV repulsion
  Eq. 11 → Composite reward (unknown-only coverage)

Observation vector per UAV (209 values):
  [row_norm, col_norm, energy_norm,          (3)   own state
   risk_w[0..99],                            (100)  risk weights
   status_norm[0..99],                       (100)  0=H, 0.5=I, 1.0=?
   Δrow_j/Δcol_j for each other UAV]        (6)   3 others × 2

Status codes:
  0 = healthy  (UAV diagnosed)
  1 = infected (UAV diagnosed)
  2 = unknown  (not yet visited)
"""

import os
import numpy as np
import pandas as pd

# ─── GRID / EPISODE CONSTANTS ─────────────────────────────────────────────────

GRID_ROWS  = 10
GRID_COLS  = 10
N_SECTORS  = GRID_ROWS * GRID_COLS   # 100
N_UAVS     = 4
T_MAX      = 72                       # episode length (days)

# ─── UAV PHYSICS ──────────────────────────────────────────────────────────────

E_MAX    = 150.0   # max battery per UAV
E_MOVE   = 1.0     # energy per move step
E_HOVER  = 1.5     # energy per STAY step
TAU_DIAG = 2       # consecutive STAY steps needed to diagnose a sector

# ─── RISK / REWARD PARAMETERS ────────────────────────────────────────────────

GAMMA      = 0.8    # detection history decay (Eq. 2)
ETA        = 0.03   # urgency growth rate for healthy sectors (Eq. 5)
                    # (lower than original — larger grid, slower urgency build-up)
ALPHA      = 0.4    # history bias in Omega (Eq. 4)
SIGMA      = 2.0    # spatial diffusion radius (larger grid → larger sigma)
H_MAX      = 10.0   # detection history saturation

PSI             = 1.0    # risk coverage weight (Eq. 11)
LAMBDA_ENG      = 0.01   # energy penalty weight (Eq. 11)
ZETA            = 0.5    # repulsion penalty weight (Eq. 11)
SIGMA_REP       = 2.0    # repulsion radius
EPSILON         = 1.0    # distance offset — keeps reward bounded (critical)
W_UNKNOWN_FLOOR = 0.3    # minimum risk weight for undiagnosed sectors
                         # prevents Omega=0 at episode start (no history, no
                         # known-infected) from killing the exploration signal

# ─── TREATMENT ────────────────────────────────────────────────────────────────
# When a UAV diagnoses an infected sector, active treatment begins immediately.
# After TREATMENT_DAYS steps the sector is marked healthy in uav_status.
# Natural healing in the simulation takes HEALING_PERIOD=14 days; UAV-triggered
# treatment takes only 3 days, making active monitoring clearly beneficial.
TREATMENT_DAYS = 3

# ─── ACTION SPACE ─────────────────────────────────────────────────────────────

ACTIONS = {
    0: ( 0,  0),   # STAY
    1: (-1,  0),   # NORTH
    2: ( 1,  0),   # SOUTH
    3: ( 0, -1),   # WEST
    4: ( 0,  1),   # EAST
}
N_ACTIONS = len(ACTIONS)

# ─── OBSERVATION SIZE ─────────────────────────────────────────────────────────
# own(3) + risk_weights(N_SECTORS) + status_norm(N_SECTORS) + other_uavs((N_UAVS-1)*2)
OBS_SIZE   = 3 + N_SECTORS + N_SECTORS + (N_UAVS - 1) * 2   # 209
JOINT_SIZE = N_UAVS * OBS_SIZE                                # 836


# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────

class UAVFieldEnv:
    """
    10×10 multi-UAV crop disease monitoring environment.

    4 UAVs start at the four corners:
      UAV 0 → (0, 0)          top-left
      UAV 1 → (0, 9)          top-right
      UAV 2 → (9, 0)          bottom-left
      UAV 3 → (9, 9)          bottom-right
    """

    def __init__(self, sim_log_path, grid_config_path, dataset_dir=None):
        """
        Args:
            sim_log_path    : path to a single simulation CSV (always required
                              as the canonical fallback / single-sim mode).
            grid_config_path: path to grid_config.json (used for neighbours).
            dataset_dir     : optional directory containing sim_XXXX.csv files
                              produced by generate_dataset.py.  When supplied,
                              ALL CSVs are loaded at init into a
                              (N_SIMS, T+1, N_SECTORS) int8 array and reset()
                              samples a random one each episode — giving the
                              agent diverse disease trajectories to generalise
                              over.  If None, only sim_log_path is used.
        """
        sim_log      = pd.read_csv(sim_log_path)
        self.T       = int(sim_log.time_step.max())

        # Pre-index simulation log → (T+1, N_SECTORS) numpy array.
        # Replaces per-step pandas filtering (O(n) → O(1) per lookup).
        sim_sorted         = sim_log.sort_values(['time_step', 'sector_id'])
        _base_lookup       = (sim_sorted['true_status']
                              .values.astype(np.int8)
                              .reshape(self.T + 1, N_SECTORS))

        # ── Dataset mode: load from .npy file or directory of CSVs ───────
        if dataset_dir is not None and dataset_dir.endswith('.npy') and os.path.isfile(dataset_dir):
            # Fast path: single pre-built numpy array (N_SIMS, T+1, N_SECTORS)
            self._all_lookups = np.load(dataset_dir).astype(np.int8)
            print(f"[UAVFieldEnv] Dataset loaded: "
                  f"{self._all_lookups.shape} int8 array "
                  f"({self._all_lookups.nbytes // 1_000_000} MB)", flush=True)
        elif dataset_dir is not None and os.path.isdir(dataset_dir):
            # Legacy path: directory of sim_XXXX.csv files
            csv_files = sorted(
                f for f in os.listdir(dataset_dir)
                if f.startswith('sim_') and f.endswith('.csv')
            )
            if csv_files:
                print(f"[UAVFieldEnv] Loading {len(csv_files)} simulations "
                      f"from {dataset_dir} ...", flush=True)
                sims = []
                for fname in csv_files:
                    df = pd.read_csv(os.path.join(dataset_dir, fname))
                    df_s = df.sort_values(['time_step', 'sector_id'])
                    arr  = (df_s['true_status']
                            .values.astype(np.int8)
                            .reshape(self.T + 1, N_SECTORS))
                    sims.append(arr)
                self._all_lookups = np.stack(sims, axis=0)
                print(f"[UAVFieldEnv] Dataset loaded: "
                      f"{self._all_lookups.shape} int8 array "
                      f"({self._all_lookups.nbytes // 1_000_000} MB)", flush=True)
            else:
                self._all_lookups = None
        else:
            self._all_lookups = None

        # status_lookup will be set / resampled in reset()
        self._base_lookup  = _base_lookup
        self.status_lookup = _base_lookup

        # sector_id → (row, col)
        self.sector_pos = {
            sid: (sid // GRID_COLS, sid % GRID_COLS)
            for sid in range(N_SECTORS)
        }
        # (row, col) → sector_id
        self.pos_to_sid = {v: k for k, v in self.sector_pos.items()}
        self.neighbors  = self._build_neighbors()

        # Pre-built position arrays for vectorized distance computation.
        self.sector_rows = np.arange(N_SECTORS, dtype=np.float32) // GRID_COLS
        self.sector_cols = np.arange(N_SECTORS, dtype=np.float32) %  GRID_COLS
        # Shape (N_SECTORS, 2) — used in omega and reward batch computations.
        self.sector_rc   = np.stack([self.sector_rows,
                                     self.sector_cols], axis=1)  # (100, 2)

        # Precomputed Gaussian kernel denominator constants.
        self._two_sigma2     = 2.0 * SIGMA     ** 2   # for omega
        self._two_sigma_rep2 = 2.0 * SIGMA_REP ** 2   # for repulsion

        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Resets to t=0. Returns initial observations for all UAVs.

        In dataset mode (dataset_dir supplied at init), samples a random
        simulation trajectory so each episode sees a different disease spread.
        """
        # Resample trajectory if dataset is loaded
        if self._all_lookups is not None:
            idx = np.random.randint(len(self._all_lookups))
            self.status_lookup = self._all_lookups[idx]
        else:
            self.status_lookup = self._base_lookup

        self.t           = 0
        self.true_status = self._load_true_status(0)
        self.uav_status  = np.full(N_SECTORS, 2, dtype=int)
        self.H           = np.zeros(N_SECTORS, dtype=float)
        self.last_visit  = np.zeros(N_SECTORS, dtype=int)

        # UAVs at four corners
        self.uav_pos = [
            (0,              0),
            (0,              GRID_COLS - 1),
            (GRID_ROWS - 1,  0),
            (GRID_ROWS - 1,  GRID_COLS - 1),
        ]
        self.energy           = [E_MAX] * N_UAVS
        self.dwell            = [0]     * N_UAVS
        # treatment_timer[sid] > 0  → sector is being treated (uav_status == 1)
        # reaches 0                 → sector heals (uav_status set to 0)
        self.treatment_timer  = np.zeros(N_SECTORS, dtype=int)
        self.w                = self._compute_risk_weights()

        return self._get_all_obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        """
        Executes one time step for all UAVs simultaneously.

        Args:
            actions : list of N_UAVS ints (0–4)

        Returns:
            obs     : list of N_UAVS observation arrays
            rewards : list of N_UAVS floats
            done    : bool
            info    : dict
        """
        assert len(actions) == N_UAVS
        energy_consumed = [0.0] * N_UAVS

        # ── Execute actions ───────────────────────────────────────────────────
        for u in range(N_UAVS):
            dr, dc   = ACTIONS[actions[u]]
            r, c     = self.uav_pos[u]
            nr, nc   = r + dr, c + dc

            if actions[u] == 0:
                self.dwell[u]      += 1
                energy_consumed[u]  = E_HOVER
            else:
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    self.uav_pos[u] = (nr, nc)
                    self.dwell[u]   = 0
                energy_consumed[u] = E_MOVE

            self.energy[u] = max(0.0, self.energy[u] - energy_consumed[u])

        # ── Diagnosis check ───────────────────────────────────────────────────
        for u in range(N_UAVS):
            if self.dwell[u] >= TAU_DIAG:
                sid = self.pos_to_sid[self.uav_pos[u]]
                if self.uav_status[sid] == 2:
                    self.uav_status[sid] = int(self.true_status[sid])
                    # Infected sector detected → start treatment countdown
                    if self.uav_status[sid] == 1:
                        self.treatment_timer[sid] = TREATMENT_DAYS

                b_kt = 1 if (self.true_status[sid] == 1 and
                             self.uav_status[sid] == 1) else 0
                self.H[sid]          = min(H_MAX, GAMMA * self.H[sid] + b_kt)
                self.last_visit[sid] = self.t

        # ── Advance time ──────────────────────────────────────────────────────
        self.t += 1
        done = (self.t >= self.T) or all(e <= 0 for e in self.energy)

        if not done:
            self.true_status = self._load_true_status(self.t)

        # ── Treatment countdown ───────────────────────────────────────────────
        # Decrement timers for all sectors under active treatment.
        active_tx = (self.uav_status == 1) & (self.treatment_timer > 0)
        self.treatment_timer[active_tx] -= 1

        # Sectors whose treatment just completed → mark as healthy.
        newly_healed = (self.uav_status == 1) & (self.treatment_timer == 0)
        if newly_healed.any():
            self.uav_status[newly_healed]      = 0
            # Override ground truth so treated sectors don't spread disease.
            self.true_status[newly_healed]     = 0

        # ── Re-infection detection ────────────────────────────────────────────
        # A sector the UAV believes is healthy (uav_status==0) but is actually
        # infected in the current ground truth must be flagged as unknown again
        # so the UAVs know to re-visit and re-treat it.
        re_infected = (self.uav_status == 0) & (self.true_status == 1)
        if re_infected.any():
            self.uav_status[re_infected]     = 2   # unknown — needs re-diagnosis
            self.treatment_timer[re_infected] = 0

        # Decay history for every sector not visited this step (vectorized).
        unvisited = (self.last_visit != self.t)
        self.H[unvisited] *= GAMMA

        self.w   = self._compute_risk_weights()
        rewards  = [self._compute_reward(u, energy_consumed[u])
                    for u in range(N_UAVS)]
        obs      = self._get_all_obs()
        info     = {
            "t":               self.t,
            "uav_pos":         list(self.uav_pos),
            "energy":          list(self.energy),
            "uav_status":      self.uav_status.copy(),
            "true_status":     self.true_status.copy(),
            "risk_weights":    self.w.copy(),
            "dwell":           list(self.dwell),
            "treatment_timer": self.treatment_timer.copy(),
        }
        return obs, rewards, done, info

    # ── Risk Computations ─────────────────────────────────────────────────────

    def _compute_risk_weights(self):
        """Eq. 5 — dynamic risk weight per sector (fully vectorized)."""
        w            = np.zeros(N_SECTORS, dtype=np.float32)
        infected_m   = (self.uav_status == 1)
        healthy_m    = (self.uav_status == 0)
        unknown_m    = (self.uav_status == 2)

        # Infected → always 1.0
        w[infected_m] = 1.0

        # Healthy → min(1, η × Δt)
        if healthy_m.any():
            delta_t       = self.t - self.last_visit[healthy_m]
            w[healthy_m]  = np.minimum(1.0, ETA * delta_t)

        # Unknown → max(W_UNKNOWN_FLOOR, min(1, Omega))
        #
        # The paper's formula min(1, Omega) collapses to 0 at episode start
        # because H=0 and K_inf is empty, giving UAVs zero exploration signal.
        # Setting the floor to W_UNKNOWN_FLOOR=0.3 solves this while keeping
        # the weight well below the infected weight of 1.0, so the EXPLORE_BONUS
        # remains the dominant incentive for diagnosis rather than proximity.
        #
        # Omega is still computed so sectors near known-infected clusters get
        # boosted above the floor (spatial risk diffusion from Eq. 4 is intact).
        if unknown_m.any():
            omega        = self._compute_omega_batch(unknown_m, infected_m)
            w[unknown_m] = np.maximum(W_UNKNOWN_FLOOR,
                                      np.minimum(1.0, omega))
        return w

    def _compute_omega_batch(self, unknown_mask, infected_mask):
        """
        Eq. 4 — computes Omega for ALL unknown sectors simultaneously.

        Replaces the per-sector loop; uses numpy broadcasting so the
        (N_unknown × N_infected) distance matrix is built in one shot.
        """
        unk_idx = np.where(unknown_mask)[0]          # (N_unk,)

        # History term — vectorized
        history = ALPHA * (self.H[unk_idx] / H_MAX)  # (N_unk,)

        inf_idx = np.where(infected_mask)[0]          # (N_inf,)
        if inf_idx.size == 0:
            return history

        # Positions: (N_unk, 1, 2) − (1, N_inf, 2) → (N_unk, N_inf, 2)
        diff    = (self.sector_rc[unk_idx, np.newaxis, :]
                   - self.sector_rc[np.newaxis, inf_idx, :])
        dist_sq = (diff ** 2).sum(axis=2)             # (N_unk, N_inf)

        spatial = np.exp(-dist_sq / self._two_sigma2).sum(axis=1)  # (N_unk,)

        return history + (1 - ALPHA) * spatial

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, u, energy_consumed):
        """
        Composite reward — coverage over KNOWN INFECTED sectors only.

        The original formulation summed w[k]/(dist+ε) over all unknown sectors,
        which created a perverse incentive: every diagnosis removed a sector from
        the reward pool, so the policy learned to never diagnose.  EXPLORE_BONUS
        (applied in the training loop) could never overcome this because the
        proximity sum dominated the total episode reward by 1000x.

        Fix: only known-infected sectors (uav_status==1, w=1.0) contribute to
        coverage.  This gives UAVs a navigation reward for staying near active
        hotspots after discovery, without punishing them for diagnosing.

        Exploration of unknowns is handled separately by a small per-step
        VISIT_BONUS (in the training loop) and a large per-diagnosis EXPLORE_BONUS,
        which together create a clean incentive: visit unknown sectors, hover to
        diagnose, collect bonus, move on.
        """
        r_u, c_u   = self.uav_pos[u]
        infected_m = (self.uav_status == 1)

        if infected_m.any():
            inf_idx  = np.where(infected_m)[0]
            dr       = self.sector_rows[inf_idx] - r_u
            dc       = self.sector_cols[inf_idx] - c_u
            dist     = np.sqrt(dr * dr + dc * dc)
            coverage = float(np.sum(self.w[inf_idx] / (dist + EPSILON)))
        else:
            coverage = 0.0

        repulsion = self._compute_repulsion(u)
        return PSI * coverage - LAMBDA_ENG * energy_consumed - ZETA * repulsion

    def _compute_repulsion(self, u):
        """Eq. 9 — inter-UAV repulsion penalty (vectorized over other UAVs)."""
        r_u, c_u  = self.uav_pos[u]
        others    = [j for j in range(N_UAVS) if j != u]
        r_others  = np.array([self.uav_pos[j][0] for j in others], dtype=np.float32)
        c_others  = np.array([self.uav_pos[j][1] for j in others], dtype=np.float32)
        dist_sq   = (r_others - r_u) ** 2 + (c_others - c_u) ** 2
        return float(np.sum(np.exp(-dist_sq / self._two_sigma_rep2)))

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs(self, u):
        """
        Returns flat observation vector for UAV u (OBS_SIZE = 209 values).

          [0:3]      own row_norm, col_norm, energy_norm
          [3:103]    risk_weight per sector
          [103:203]  uav_status normalised (0=H, 0.5=I, 1.0=?)
          [203:209]  Δrow/Δcol for each other UAV (3 × 2)
        """
        r_u, c_u = self.uav_pos[u]

        own = np.array([
            r_u / (GRID_ROWS - 1),
            c_u / (GRID_COLS - 1),
            self.energy[u] / E_MAX,
        ], dtype=np.float32)

        risk        = self.w.astype(np.float32)
        status_norm = (self.uav_status / 2.0).astype(np.float32)

        other_pos = []
        for j in range(N_UAVS):
            if j == u:
                continue
            r_j, c_j = self.uav_pos[j]
            other_pos.append((r_j - r_u) / (GRID_ROWS - 1))
            other_pos.append((c_j - c_u) / (GRID_COLS - 1))
        other_pos = np.array(other_pos, dtype=np.float32)

        return np.concatenate([own, risk, status_norm, other_pos])

    def _get_all_obs(self):
        return [self._get_obs(u) for u in range(N_UAVS)]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_true_status(self, t):
        """O(1) lookup into pre-indexed status array (no pandas at runtime)."""
        return self.status_lookup[t].copy()

    def _build_neighbors(self):
        neighbors = {}
        for sid in range(N_SECTORS):
            r, c  = self.sector_pos[sid]
            nbrs  = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                        nbrs.append(self.pos_to_sid[(nr, nc)])
            neighbors[sid] = nbrs
        return neighbors

    def get_grid_summary(self):
        """Returns a compact ASCII summary of the current grid state."""
        pos_set = {self.uav_pos[u]: u for u in range(N_UAVS)}
        lines   = [f"\nt={self.t}  "
                   + "  ".join(f"UAV{u}@{self.uav_pos[u]} E={self.energy[u]:.0f}"
                                for u in range(N_UAVS))]
        header  = "     " + "".join(f"{c:3}" for c in range(GRID_COLS))
        lines.append(header)
        for r in range(GRID_ROWS):
            row_str = f"r{r:2}  "
            for c in range(GRID_COLS):
                sid = self.pos_to_sid[(r, c)]
                sym = ["H", "I", "?"][self.uav_status[sid]]
                if (r, c) in pos_set:
                    sym = str(pos_set[(r, c)])
                row_str += f"{sym:>3}"
            lines.append(row_str)
        return "\n".join(lines)
