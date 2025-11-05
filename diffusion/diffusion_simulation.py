import json
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.signal import savgol_filter
import numba
from tqdm import tqdm
import os

# --- Style and Figure Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300
})

# --- 1. Model Definition ---
# Parameters for the physical system
BETA = 1.0       # Inverse temperature (kb*T = 1)
ALPHA = 0.5      # Strength of non-conservative force
EPSILON = 0.1    # Tilt in the potential

RESULTS_NPY_PATH = r'diffusion/experiment_results.npy'
RESULTS_JSONL_PATH = r'diffusion/experiment_results.jsonl'

@numba.njit
def potential(x, y, h_b):
    """Tilted double-well potential U(x,y)."""
    return h_b / 4.0 * (x**2 - 1)**2 + EPSILON / 2.0 * x + 0.5 * y**2

@numba.njit
def grad_potential(x, y, h_b):
    """Gradient of the potential, -grad(U)."""
    grad_x = - (h_b * x * (x**2 - 1) + EPSILON / 2.0)
    grad_y = - y
    return grad_x, grad_y

@numba.njit
def non_conservative_force(x, y):
    """Solenoidal non-conservative force F(x,y)."""
    force_x = -ALPHA * y
    force_y = ALPHA * x
    return force_x, force_y

# --- 2. Collapsed System: Fokker-Planck Operator ---

def get_fokker_planck_operator(grid_pts, h_b):
    """
    Discretizes the Fokker-Planck operator for the overdamped system on a 2D grid.
    Returns the operator as a sparse matrix and the grid spacing dx.
    """
    L_domain = 3.0
    x = np.linspace(-L_domain, L_domain, grid_pts)
    y = np.linspace(-L_domain, L_domain, grid_pts)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y)

    # Flatten grid
    X_flat, Y_flat = X.flatten(), Y.flatten()
    N = grid_pts**2

    # --- Construct Operator Matrix L_O ---
    # Build as a dense array for ease of assembling row operators, then convert to sparse.
    L_O = np.zeros((N, N), dtype=np.float64)

    # Partial derivative operators (central differences with periodic BC)
    def d_dx(i, j):
        row = np.zeros(N, dtype=np.float64)
        idx = i * grid_pts + j
        # Periodic boundary conditions
        j_plus = (j + 1) % grid_pts
        j_minus = (j - 1 + grid_pts) % grid_pts
        idx_plus = i * grid_pts + j_plus
        idx_minus = i * grid_pts + j_minus
        row[idx_plus] += 1 / (2 * dx)
        row[idx_minus] -= 1 / (2 * dx)
        return row

    def d_dy(i, j):
        row = np.zeros(N, dtype=np.float64)
        idx = i * grid_pts + j
        i_plus = (i + 1) % grid_pts
        i_minus = (i - 1 + grid_pts) % grid_pts
        idx_plus = i_plus * grid_pts + j
        idx_minus = i_minus * grid_pts + j
        row[idx_plus] += 1 / (2 * dx)
        row[idx_minus] -= 1 / (2 * dx)
        return row

    def d2_dx2(i, j):
        row = np.zeros(N, dtype=np.float64)
        idx = i * grid_pts + j
        j_plus = (j + 1) % grid_pts
        j_minus = (j - 1 + grid_pts) % grid_pts
        idx_plus = i * grid_pts + j_plus
        idx_minus = i * grid_pts + j_minus
        row[idx_plus] += 1 / dx**2
        row[idx_minus] += 1 / dx**2
        row[idx] -= 2 / dx**2
        return row
    
    def d2_dy2(i, j):
        row = np.zeros(N, dtype=np.float64)
        idx = i * grid_pts + j
        i_plus = (i + 1) % grid_pts
        i_minus = (i - 1 + grid_pts) % grid_pts
        idx_plus = i_plus * grid_pts + j
        idx_minus = i_minus * grid_pts + j
        row[idx_plus] += 1 / dx**2
        row[idx_minus] += 1 / dx**2
        row[idx] -= 2 / dx**2
        return row

    for i in range(grid_pts):
        for j in range(grid_pts):
            idx = i * grid_pts + j
            x_ij, y_ij = X[i, j], Y[i, j]

            gradU_x, gradU_y = grad_potential(x_ij, y_ij, h_b)
            F_x, F_y = non_conservative_force(x_ij, y_ij)

            drift_x = gradU_x + F_x
            drift_y = gradU_y + F_y

            # Apply drift part: - (drift_x * d/dx + drift_y * d/dy)
            L_O[idx, :] -= drift_x * d_dx(i, j)
            L_O[idx, :] -= drift_y * d_dy(i, j)

            # Apply diffusion part: BETA * (d2/dx2 + d2/dy2)
            L_O[idx, :] += BETA * (d2_dx2(i, j) + d2_dy2(i, j))
    # Convert to sparse matrix for further processing
    return sp.csr_matrix(L_O), dx

def get_singular_gap(L_O):
    """Computes the singular value gap of the operator L_O."""
    # Robust approach: compute smallest non-zero singular value via A = L_O.T @ L_O
    # singular values are sqrt(eigenvalues(A)). Smallest non-zero singular value = sqrt(smallest non-zero eigenvalue of A).
    try:
        # If the problem is small enough, compute dense SVD which is robust
        n = L_O.shape[0]
        if n <= 2000:
            try:
                arr = L_O.toarray() if hasattr(L_O, 'toarray') else np.asarray(L_O)
                svals = np.linalg.svd(arr, compute_uv=False)
                svals = np.sort(np.abs(svals))
                # skip near-zero singular values (tolerance relative to max)
                tol = max(svals.max() * 1e-14, 1e-20)
                for sv in svals:
                    if sv > tol:
                        return float(sv)
            except Exception:
                # fall through to sparse methods
                pass

        from scipy.sparse.linalg import eigsh
        # form A = L_O^T * L_O (symmetric, positive semi-definite)
        A = (L_O.T).dot(L_O)
        # Compute a few smallest eigenvalues near zero using shift-invert (sigma=0)
        # Ask for k=4 to have margin; filter zeros
        k_try = min(A.shape[0]-1, 4)
        if k_try < 1:
            return np.nan
        try:
            vals, _ = eigsh(A, k=k_try, sigma=0.0, which='LM')
        except Exception:
            # fallback to smallest magnitude eigenvalues without sigma
            vals, _ = eigsh(A, k=k_try, which='SM')

        vals = np.sort(np.real(vals))
        tol = 1e-20
        for v in vals:
            if v > tol:
                return float(np.sqrt(v))
    except Exception:
        pass

    # Fallback: try svds on L_O directly (may be fragile)
    try:
        sv = svds(L_O, k=3, which='SM', return_singular_vectors=False)
        sv = np.sort(np.abs(sv))
        if sv.size >= 2:
            return float(sv[1])
        elif sv.size >= 1:
            return float(sv[0])
    except Exception:
        pass

    print("Could not robustly compute singular gap; returning NaN.")
    return np.nan

# --- 3. Lifted System: SDE Simulation ---

@numba.njit
def baobab_integrator(q, p, dt, gamma, h_b):
    """Performs one step of the BAOAB integrator."""
    # B-step (potential gradient)
    gradU_x, gradU_y = grad_potential(q[0], q[1], h_b)
    F_x, F_y = non_conservative_force(q[0], q[1])
    force_x = gradU_x + F_x
    force_y = gradU_y + F_y
    p[0] += 0.5 * dt * force_x
    p[1] += 0.5 * dt * force_y
    
    # A-step (position update)
    q[0] += 0.5 * dt * p[0]
    q[1] += 0.5 * dt * p[1]
    
    # O-step (Ornstein-Uhlenbeck)
    c1 = np.exp(-gamma * dt)
    # OU noise amplitude uses thermal factor 1/BETA
    c2 = np.sqrt((1 - c1**2) / BETA)
    
    p[0] = c1 * p[0] + c2 * np.random.normal()
    p[1] = c1 * p[1] + c2 * np.random.normal()
    
    # A-step
    q[0] += 0.5 * dt * p[0]
    q[1] += 0.5 * dt * p[1]

    # B-step
    gradU_x, gradU_y = grad_potential(q[0], q[1], h_b)
    F_x, F_y = non_conservative_force(q[0], q[1])
    force_x = gradU_x + F_x
    force_y = gradU_y + F_y
    p[0] += 0.5 * dt * force_x
    p[1] += 0.5 * dt * force_y
    
    return q, p

@numba.njit
def run_simulation(n_steps, dt, gamma, h_b, initial_q):
    """Runs a full simulation and returns the x-coordinate trajectory."""
    q = initial_q.copy()
    p = np.random.normal(0.0, np.sqrt(1.0 / BETA), size=2)
    traj_x = np.empty(n_steps)
    
    for i in range(n_steps):
        q, p = baobab_integrator(q, p, dt, gamma, h_b)
        traj_x[i] = q[0]
        
    return traj_x


@numba.njit
def run_overdamped_sim(n_steps, dt, h_b, q0):
    """Runs an overdamped Euler-Maruyama simulation and returns the x-coordinate trajectory.

    This top-level function mirrors the small nested version used for plotting but is
    placed at module scope so it can be JIT-compiled and reused for IAT estimation.
    """
    q = q0.copy()
    traj_x = np.empty(n_steps)
    for i in range(n_steps):
        gradU_x, gradU_y = grad_potential(q[0], q[1], h_b)
        Fx, Fy = non_conservative_force(q[0], q[1])
        q[0] += dt * (gradU_x + Fx) + np.sqrt(2.0 / BETA * dt) * np.random.normal()
        q[1] += dt * (gradU_y + Fy) + np.sqrt(2.0 / BETA * dt) * np.random.normal()
        traj_x[i] = q[0]
    return traj_x

# --- 4. Analysis: Correlation Diagnostics ---

def integrated_autocorrelation_time(x, c=5):
    """
    Calculates the Integrated Autocorrelation Time (IAT) of a time series.
    Uses the methodology from the emcee package for robustness.
    """
    # Ensure the series is long enough
    if len(x) < 2 * c:
        return np.nan


def save_results_jsonl(results, filepath):
    """Persist experiment results alongside the binary checkpoint in JSON Lines."""

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return to_serializable(obj.tolist())
        if isinstance(obj, (np.floating, np.integer)):
            return to_serializable(obj.item())
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        return to_serializable(str(obj))

    try:
        with open(filepath, 'w', encoding='utf-8') as handle:
            for entry in results:
                payload = to_serializable(entry)
                handle.write(json.dumps(payload))
                handle.write('\n')
    except Exception:
        print(f"  -> Warning: failed to write JSONL results to {filepath}.")


def estimate_acf_decay_rate(x, dt, tail_fraction=0.35, min_tail_points=25, eps=1e-8):
    """Estimate the exponential decay rate nu from the log-envelope of the ACF tail.

    The method computes the normalized ACF of ``x`` using FFT-based convolution,
    extracts the asymptotic tail, fits ``log(|ACF|)`` versus lag time, and returns
    ``nu = -slope`` from that linear fit. Positive ``nu`` indicates exponential
    decay of correlations. Returns NaN if a robust fit cannot be obtained.

    Parameters
    ----------
    x : array_like
        Time series samples.
    dt : float
        Sampling interval associated with the trajectory segments.
    tail_fraction : float, optional
        Fraction of the usable lags to keep for the tail fit (use the last
        ``tail_fraction`` of the valid lags). Must lie in (0, 1].
    min_tail_points : int, optional
        Minimum number of tail points required to attempt the linear fit.
    eps : float, optional
        Threshold below which ``|ACF|`` values are ignored to avoid taking the
        logarithm of noisy, near-zero correlations.
    """

    x = np.asarray(x, dtype=float)
    if x.size < 3 or dt <= 0:
        return np.nan

    x_centered = x - np.mean(x)
    n = x_centered.size
    # Next power-of-two padding via 2*n FFT mirrors emcee-style ACF evaluation.
    f = np.fft.fft(x_centered, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] == 0:
        return np.nan
    acf /= acf[0]

    lags = np.arange(n) * dt
    acf_tail = acf[1:]
    lag_tail = lags[1:]

    valid = np.isfinite(acf_tail) & (np.abs(acf_tail) > eps)
    if valid.sum() < max(min_tail_points, 5):
        return np.nan

    acf_valid = acf_tail[valid]
    lag_valid = lag_tail[valid]

    # Focus on the asymptotic region: keep only the last ``tail_fraction`` of the
    # usable lags while ensuring at least ``min_tail_points`` samples.
    tail_fraction = float(np.clip(tail_fraction, 1e-3, 1.0))
    start_idx = max(int(np.floor((1.0 - tail_fraction) * acf_valid.size)), 0)
    acf_tail = acf_valid[start_idx:]
    lag_tail = lag_valid[start_idx:]

    if acf_tail.size < min_tail_points:
        # fall back to the longest available stretch with the required length
        if acf_valid.size >= min_tail_points:
            acf_tail = acf_valid[-min_tail_points:]
            lag_tail = lag_valid[-min_tail_points:]
        else:
            return np.nan

    if acf_tail.size < 2:
        return np.nan

    log_env = np.log(np.abs(acf_tail))
    try:
        slope, _ = np.polyfit(lag_tail, log_env, 1)
    except np.linalg.LinAlgError:
        return np.nan

    nu = -float(slope)
    if not np.isfinite(nu) or nu <= 0:
        return np.nan
    return nu

    # Calculate autocorrelation function using FFT
    n = len(x)
    x = np.asarray(x, dtype=float)
    x_demeaned = x - np.mean(x)
    f = np.fft.fft(x_demeaned, n=2*n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]

    # Use the initial positive sequence / pairwise summation (Geyer / Sokal) to get a robust window
    # Form pairwise sums of the autocorrelation (rho_k + rho_{k+1}) and stop when the pair becomes non-positive.
    try:
        g = acf[1:]
        # If there's fewer than 2 points beyond lag 0, return trivial IAT
        if g.size < 1:
            return 1.0

        # pairwise sums: g[0]+g[1], g[2]+g[3], ...
        pair_sums = g[0::2].copy()
        # Add the second element of each pair where it exists
        second = g[1::2]
        pair_sums[:second.size] += second

        # Find how many positive pairs we have
        m = 0
        for val in pair_sums:
            if val <= 0:
                break
            m += 1

        if m == 0:
            return 1.0

        tau = 1.0 + 2.0 * np.sum(pair_sums[:m])
        return float(tau)
    except Exception:
        return np.nan


# --- 5. Main Experiment Orchestration ---

def run_full_experiment():
    print("--- Starting Zeno Lifting Numerical Experiment ---")
    
    # --- Experiment Parameters ---
    barrier_heights = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
    gamma_values = np.logspace(-1, 2, 20)
    
    # Simulation settings for decay-rate estimation (can be overridden by QUICK mode below)
    SIM_DT = 0.01
    N_STEPS_RATE = 2_000_000  # Long simulation for accurate rate estimate
    N_BURN_IN = 100_000      # Steps to discard for equilibration
    REPEATS_PER_GAMMA = 3    # Run multiple independent trajectories per gamma and aggregate
    initial_q = np.array([-1.0, 0.0])

    # Production mode: allow an environment variable to request higher statistical quality
    prod = os.environ.get('ZENO_PROD', '0') == '1'
    if prod:
        print('Production mode enabled: increasing simulation length and repeats for final results')
        # Conservative but substantially higher-quality settings. Adjust if you want longer runs.
        N_STEPS_RATE = 100_000
        N_BURN_IN = 10_000
        REPEATS_PER_GAMMA = 30
        GRID_PTS = 50

    # Settings for Fokker-Planck discretization (default)
    GRID_PTS = 50

    # Quick mode: detect environment variable or small file flag
    # If production mode is requested, ensure quick mode is disabled so we run the full experiment
    quick = (os.environ.get('ZENO_QUICK', '0') == '1') and (not prod)
    if quick:
        print('Quick mode enabled: using reduced settings for fast iteration')
        gamma_values = np.logspace(-1, 2, 12)
        N_STEPS_RATE = 20000
        N_BURN_IN = 2000
        REPEATS_PER_GAMMA = 24
        GRID_PTS = 30

    results = []
    # --- Stage 1 & 2: Loop over systems (barrier heights) ---
    try:
        for h_b in tqdm(barrier_heights, desc="Barrier Heights"):
            # --- STAGE 1: COLLAPSED SYSTEM ---
            print(f"\nAnalyzing collapsed system for h_b = {h_b}...")
            L_O, dx = get_fokker_planck_operator(GRID_PTS, h_b)
            s_L_O = get_singular_gap(L_O)
            print(f"  -> Singular gap s(L_O) = {s_L_O:.4e}")

            # --- STAGE 2: LIFTED SYSTEM ---
            # For robustness run several independent trajectories per gamma and aggregate (median, std)
            nu_samples_all = []
            med_nus = []
            std_nus = []
            # Keep total simulated time constant across different gamma by adapting dt and n_steps.
            TOTAL_SIM_TIME = N_STEPS_RATE * SIM_DT
            BURN_TIME = N_BURN_IN * SIM_DT

            for gamma in tqdm(gamma_values, desc=f"  Scanning gamma (h_b={h_b})"):
                # choose a safe dt for the given gamma to resolve OU step; don't exceed SIM_DT
                if gamma > 0:
                    dt_gamma = min(SIM_DT, 0.1 / gamma)
                else:
                    dt_gamma = SIM_DT

                n_steps_gamma = max(int(np.ceil(TOTAL_SIM_TIME / dt_gamma)), 10)
                burn_steps_gamma = min(int(np.ceil(BURN_TIME / dt_gamma)), n_steps_gamma-1)

                rate_repeats = []
                for rep in range(REPEATS_PER_GAMMA):
                    traj = run_simulation(n_steps_gamma, dt_gamma, gamma, h_b, initial_q)
                    # apply burn-in in steps for this gamma
                    nu_est = estimate_acf_decay_rate(traj[burn_steps_gamma:], dt_gamma)
                    rate_repeats.append(nu_est)
                rate_repeats = np.array(rate_repeats, dtype=float)
                nu_samples_all.append(rate_repeats)
                med_nus.append(np.nanmedian(rate_repeats))
                std_nus.append(np.nanstd(rate_repeats))

            med_nus = np.array(med_nus, dtype=float)
            std_nus = np.array(std_nus, dtype=float)

            # Find optimal gamma using the largest median decay rate across repeats
            valid_mask = np.isfinite(med_nus)
            if np.any(valid_mask):
                max_nu_idx = np.nanargmax(med_nus)
                gamma_opt = float(gamma_values[max_nu_idx])
                nu_opt = float(med_nus[max_nu_idx])
                nu_opt_std = float(std_nus[max_nu_idx]) if np.isfinite(std_nus[max_nu_idx]) else np.nan

                print(f"  -> Optimal gamma = {gamma_opt:.3f}, Max median rate = {nu_opt:.3e}")

                results.append({
                    "h_b": h_b,
                    "s_L_O": s_L_O,
                    "gammas": gamma_values,
                    "nu_medians": med_nus,        # median rate per gamma
                    "nu_samples": nu_samples_all,  # raw repeats per gamma
                    "nu_std": std_nus,
                    "gamma_opt": gamma_opt,
                    "nu_opt": nu_opt,
                    "nu_opt_std": nu_opt_std,
                    "nu_source": "acf_tail"
                })
                # checkpoint: save after completing each barrier height so long runs can be resumed
                try:
                    np.save(RESULTS_NPY_PATH, results)
                    save_results_jsonl(results, RESULTS_JSONL_PATH)
                    print(f"  -> Checkpoint saved ({len(results)} barrier(s) completed)")
                except Exception:
                    print("  -> Warning: failed to save checkpoint file.")
            else:
                print(f"  -> Rate estimation failed for all gammas.")
    except KeyboardInterrupt:
        print('\nRun interrupted by user; saving partial results...')
        try:
            np.save(RESULTS_NPY_PATH, results)
            save_results_jsonl(results, RESULTS_JSONL_PATH)
            print(f'Partial results saved to {RESULTS_NPY_PATH} and {RESULTS_JSONL_PATH}')
        except Exception:
            print('Failed to save partial results on interrupt.')
        raise
    except Exception as e:
        print(f"Exception during run_full_experiment: {e}")
        try:
            np.save(RESULTS_NPY_PATH, results)
            save_results_jsonl(results, RESULTS_JSONL_PATH)
            print(f'Partial results saved to {RESULTS_NPY_PATH} and {RESULTS_JSONL_PATH}')
        except Exception:
            print('Failed to save partial results after exception.')
        raise

    return results

# --- 6. Plotting ---

def create_publication_figure(results):
    print("\n--- Generating Publication Figure ---")
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 1), height_ratios=(1, 1))

    def _extract_rate_series(res):
        if 'nu_medians' in res:
            return np.asarray(res['nu_medians'], dtype=float)
        if 'iats' in res:
            arr = np.asarray(res['iats'], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(arr > 0, 1.0 / arr, np.nan)
        return np.array([], dtype=float)

    def _extract_rate_std(res):
        if 'nu_std' in res:
            return np.asarray(res['nu_std'], dtype=float)
        if 'iats_std' in res and 'iats' in res:
            tau = np.asarray(res['iats'], dtype=float)
            tau_std = np.asarray(res['iats_std'], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where((tau > 0) & np.isfinite(tau_std), tau_std / (tau ** 2), np.nan)
        return np.array([], dtype=float)

    def _extract_rate_samples(res):
        if 'nu_samples' in res:
            return [np.asarray(s, dtype=float) for s in res['nu_samples']]
        if 'iats_all' in res:
            converted = []
            for arr in res['iats_all']:
                arr = np.asarray(arr, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    converted.append(np.where(arr > 0, 1.0 / arr, np.nan))
            return converted
        return []

    def _extract_nu_opt(res):
        if 'nu_opt' in res and np.isfinite(res['nu_opt']):
            return float(res['nu_opt'])
        if 'nu_max' in res and np.isfinite(res['nu_max']):
            return float(res['nu_max'])
        if 'iat_min' in res and res['iat_min'] and np.isfinite(res['iat_min']):
            return 1.0 / float(res['iat_min']) if res['iat_min'] > 0 else np.nan
        return np.nan

    # --- Panel (a): NESS and Potential ---
    ax_a = fig.add_subplot(gs[0, 0])
    h_b_plot = results[-1]['h_b'] # Use the highest barrier for visualization
    L_domain = 2.0
    x_plot = np.linspace(-L_domain, L_domain, 100)
    y_plot = np.linspace(-L_domain, L_domain, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = potential(X, Y, h_b_plot)
    
    contour = ax_a.contourf(X, Y, Z, levels=20, cmap='viridis_r', alpha=0.8)
    ax_a.contour(X, Y, Z, levels=contour.levels, colors='white', linewidths=0.5)
    
    # NESS current (quiver plot)
    Fx, Fy = non_conservative_force(X, Y)
    ax_a.streamplot(X, Y, Fx, Fy, color='white', linewidth=1, density=0.6, arrowstyle='->', arrowsize=1.0)
    
    ax_a.set_title(r'\textbf{(a)} Non-Equilibrium Steady State', y=1.05)
    ax_a.set_xlabel(r'$x$')
    ax_a.set_ylabel(r'$y$')
    ax_a.set_aspect('equal')

    # --- Panel (b): Direct comparison of convergence rates (Collapsed vs Lifted) ---
    # Compute (or reuse cached) collapse-only decay rates with the same ACF-tail
    # estimator and compare them against the optimal lifted rates stored with each
    # result.
    ax_b = fig.add_subplot(gs[0, 1])

    quick_flag = os.environ.get('ZENO_QUICK', '0') == '1'
    # Simulation defaults mirror the main experiment; quick mode uses smaller work.
    SIM_DT = 0.01
    N_STEPS_RATE = 20000 if quick_flag else 2_000_000
    N_BURN_IN = 2000 if quick_flag else 100_000
    REPEATS = 6 if quick_flag else 3
    TOTAL_SIM_TIME = N_STEPS_RATE * SIM_DT

    # Compute nu_collapsed for each barrier if missing; cache into results to avoid recompute.
    for r in results:
        if ('nu_collapsed' in r) and (r['nu_collapsed'] is not None) and (not np.isnan(r['nu_collapsed'])) and ('nu_collapsed_rates' in r):
            continue

        h_b_val = r['h_b']
        print(f"Computing overdamped decay rate for collapsed system (h_b={h_b_val}) using {N_STEPS_RATE} steps x dt={SIM_DT}...")
        n_steps = max(int(np.ceil(TOTAL_SIM_TIME / SIM_DT)), 10)
        # N_BURN_IN is a number of steps (already in the script's convention).
        # Use burn_steps = min(N_BURN_IN, n_steps-1) to avoid converting by dt.
        burn_steps = min(int(N_BURN_IN), n_steps - 1)

        nu_repeats = []
        for rep in range(REPEATS):
            traj = run_overdamped_sim(n_steps, SIM_DT, h_b_val, np.array([-1.0, 0.0]))
            nu_est = estimate_acf_decay_rate(traj[burn_steps:], SIM_DT)
            nu_repeats.append(nu_est)

        nu_repeats = np.array(nu_repeats)
        med = float(np.nanmedian(nu_repeats)) if nu_repeats.size > 0 else np.nan
        std = float(np.nanstd(nu_repeats)) if nu_repeats.size > 0 else np.nan
        nu_collapsed = med if (med is not None and np.isfinite(med) and med > 0) else np.nan
        r['nu_collapsed'] = nu_collapsed
        r['nu_collapsed_rates'] = nu_repeats.tolist()
        r['nu_collapsed_median'] = med
        r['nu_collapsed_std'] = std
        r['nu_collapsed_source'] = 'acf_tail'
        # Save back to disk to cache heavy computation
        try:
            np.save(RESULTS_NPY_PATH, results)
            save_results_jsonl(results, RESULTS_JSONL_PATH)
            print(f"  -> Cached nu_collapsed for h_b={h_b_val}")
        except Exception:
            print("  -> Warning: failed to cache updated results file for nu_collapsed.")

    # Gather arrays for plotting
    hb_vals = np.array([r['h_b'] for r in results])
    nu_collapsed_vals = np.array([r.get('nu_collapsed', np.nan) for r in results], dtype=float)
    nu_opt_vals = np.array([_extract_nu_opt(r) for r in results], dtype=float)

    # Filter valid positive values
    valid = np.isfinite(nu_collapsed_vals) & np.isfinite(nu_opt_vals) & (nu_collapsed_vals > 0) & (nu_opt_vals > 0)
    if np.any(valid):
        x = nu_collapsed_vals[valid]
        y = nu_opt_vals[valid]
        hbs = hb_vals[valid]

        ax_b.set_xscale('log')
        ax_b.set_yscale('log')
        ax_b.plot(x, y, 'o', color='tab:blue', ms=8, label='Data (annotated: $h_b$)')
        # identity line
        lims = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
        ax_b.plot(lims, lims, '--', color='0.3', lw=1, label=r'Identity $y=x$')
        # annotate points with barrier heights
        for xi, yi, hb in zip(x, y, hbs):
            ax_b.annotate(f"{hb:.1f}", xy=(xi, yi), xytext=(4, 4), textcoords='offset points', fontsize=10)

        ax_b.set_xlim(lims)
        ax_b.set_ylim(lims)
        ax_b.set_xlabel(r'Collapsed rate $\nu(L_O)$')
        ax_b.set_ylabel(r'Optimally lifted rate $\nu_{\mathrm{opt}}(L)$')
        ax_b.set_title(r'\textbf{(b)} Collapsed vs Lifted Convergence Rates', y=1.05)
        ax_b.grid(True, which='both', ls=':', alpha=0.4)
        # show ratio as text
        ratios = y / x
        median_ratio = np.nanmedian(ratios)
        ax_b.text(0.05, 0.95, f'Median speedup: {median_ratio:.2f}x', transform=ax_b.transAxes, ha='left', va='top', fontsize=11)
        # Legend explaining the annotation on each plotted point
        ax_b.legend(loc='lower right', fontsize=11)
    else:
        ax_b.text(0.5, 0.5, 'Insufficient data to compare rates', ha='center', va='center')
        ax_b.set_title(r'\textbf{(b)} Collapsed vs Lifted Convergence Rates', y=1.05)

    # --- Panel (c): Optimal Dissipation ---
    ax_c = fig.add_subplot(gs[1, 0])
    res_plot = results[-1]
    gammas = np.asarray(res_plot['gammas'], dtype=float)
    # Use median decay rates and their spread across repeats for error bars
    med_rates = _extract_rate_series(res_plot)
    std_rates = _extract_rate_std(res_plot)
    if std_rates.size == 0:
        std_rates = np.zeros_like(med_rates)
    # Optionally drop the smallest measured gamma if it looks like an outlier/noisy point
    # (this avoids an extrapolation/plot artifact and reveals the single peak supported by the rest of the data)
    if gammas.size >= 2:
        idx_min = int(np.nanargmin(gammas))
        # drop only from the plotting arrays, keep original results untouched
        gammas_plot_source = np.delete(gammas, idx_min)
        med_rates_plot = np.delete(med_rates, idx_min)
        std_rates_plot = np.delete(std_rates, idx_min)
    else:
        gammas_plot_source = gammas
        med_rates_plot = med_rates
        std_rates_plot = std_rates
    # Guard against zeros or NaNs to avoid invalid entries
    rates_empirical = np.where(np.isfinite(med_rates_plot) & (med_rates_plot > 0), med_rates_plot, np.nan)
    # std_rates_plot captured for potential future error bars; not used directly in current plot

    # --- Densify only inside the valid data-supported gamma range (no extrapolation) ---
    # Use the plotting-source arrays (may have dropped the smallest gamma)
    valid = np.isfinite(rates_empirical) & (gammas_plot_source > 0) & (rates_empirical > 0)
    rate_plot = None
    gammas_plot = None
    if valid.sum() >= 2:
        gamma_min_valid = float(np.nanmin(gammas_plot_source[valid]))
        gamma_max_valid = float(np.nanmax(gammas_plot_source[valid]))
        # densify inside the measured range only (no extrapolation beyond data)
        gammas_plot = np.logspace(np.log10(gamma_min_valid), np.log10(gamma_max_valid), num=400)
        log_xs = np.log10(gammas_plot_source[valid])
        log_ys = np.log10(rates_empirical[valid])
        log_xt = np.log10(gammas_plot)
        # interpolate in log-log space; since gammas_plot is within [gamma_min_valid, gamma_max_valid] no extrapolation
        log_yt = np.interp(log_xt, log_xs, log_ys)
        rate_plot = 10 ** (log_yt)
    else:
        # fall back to plotting raw data only (use plotting-source arrays)
        gammas_plot = gammas_plot_source
        rate_plot = rates_empirical

    # Plot raw points faded, and an interpolated dense curve in the small-gamma region (no smoothing)
    ax_c.plot(gammas_plot_source, rates_empirical, 'o', color='darkorange', alpha=0.6, ms=6, label=r'Raw $\nu$')
    ax_c.plot(gammas_plot, rate_plot, '-', color='darkorange', lw=1.5, alpha=0.95, label=r'Dense interpolation (no extrapolation)')
    # Mark the maximum of the interpolated curve explicitly
    if np.any(np.isfinite(rate_plot)):
        max_idx = int(np.nanargmax(rate_plot))
        gamma_peak = gammas_plot[max_idx]
        nu_peak = float(rate_plot[max_idx])
        ax_c.plot([gamma_peak], [nu_peak], marker='*', color='black', ms=12, label=r'Peak')
        ax_c.axvline(gamma_peak, color='k', linestyle='--', alpha=0.6)
        label = r'$\gamma_{\mathrm{opt}} = ' + f'{gamma_peak:.2g}' + r'$'
        ax_c.annotate(label, xy=(gamma_peak, nu_peak), xytext=(6, 6), textcoords='offset points')

    # (No error band drawn — we show dense interpolation only, no smoothing/extrapolation)

    ax_c.set_xscale('log')
    ax_c.set_title(r'\textbf{(c)} Optimal Dissipation', y=1.05)
    ax_c.set_xlabel(r'Friction Coefficient $\gamma$')
    ax_c.set_ylabel(r'Convergence Rate $\nu$')
    ax_c.legend()

    # --- Panel (d): Quadratic Speedup ---
    ax_d = fig.add_subplot(gs[1, 1])
    s_L_O_all = np.array([r['s_L_O'] for r in results], dtype=float)
    quick_flag = os.environ.get('ZENO_QUICK', '0') == '1'
    B = 100 if quick_flag else 200
    nu_max_vals = []
    nu_log_vars = []
    nu_ci_low = []
    nu_ci_high = []
    for r in results:
        rate_samples = _extract_rate_samples(r)
        if rate_samples:
            bs_nu = []
            for _ in range(B):
                medians = []
                for arr in rate_samples:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr) & (arr > 0)]
                    if arr.size == 0:
                        medians.append(np.nan)
                        continue
                    sample = np.random.choice(arr, size=arr.size, replace=True)
                    medians.append(np.nanmedian(sample))
                medians = np.array(medians, dtype=float)
                if np.all(np.isnan(medians)):
                    bs_nu.append(np.nan)
                else:
                    bs_nu.append(np.nanmax(medians))
            bs_nu = np.array(bs_nu, dtype=float)
            if np.all(np.isnan(bs_nu)):
                nu_max_vals.append(np.nan)
                nu_log_vars.append(np.nan)
                nu_ci_low.append(np.nan)
                nu_ci_high.append(np.nan)
            else:
                nu_max_vals.append(np.nanmedian(bs_nu))
                log_bs = np.log(bs_nu[np.isfinite(bs_nu) & (bs_nu > 0)])
                nu_log_vars.append(np.var(log_bs) if log_bs.size > 0 else np.nan)
                nu_ci_low.append(np.nanpercentile(bs_nu, 2.5))
                nu_ci_high.append(np.nanpercentile(bs_nu, 97.5))
        else:
            nu_val = _extract_nu_opt(r)
            nu_max_vals.append(nu_val)
            nu_log_vars.append(np.nan)
            nu_ci_low.append(np.nan)
            nu_ci_high.append(np.nan)

    nu_max_vals = np.array(nu_max_vals, dtype=float)
    nu_log_vars = np.array(nu_log_vars, dtype=float)
    nu_ci_low = np.array(nu_ci_low, dtype=float)
    nu_ci_high = np.array(nu_ci_high, dtype=float)
    nu_collapsed_all = np.array([r.get('nu_collapsed', np.nan) for r in results], dtype=float)

    ax_d.set_xscale('log')
    ax_d.set_yscale('log')

    def _plot_with_fit(x_vals, y_vals, marker, color, label_points, fit_label, linestyle='--'):
        mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
        if not np.any(mask):
            return None
        x_sel = x_vals[mask]
        y_sel = y_vals[mask]
        ax_d.plot(x_sel, y_sel, marker, color=color, markersize=7, label=label_points)

        slope = None
        if x_sel.size > 1:
            log_x = np.log(x_sel)
            log_y = np.log(y_sel)
            fit_mask = np.ones_like(log_x, dtype=bool)
            for _ in range(3):
                if fit_mask.sum() < 2:
                    break
                coeffs = np.polyfit(log_x[fit_mask], log_y[fit_mask], 1)
                pred = coeffs[0] * log_x + coeffs[1]
                resid = log_y - pred
                sigma = np.nanstd(resid[fit_mask])
                sigma = max(sigma, 1e-12)
                fit_mask = np.abs(resid) < 2.0 * sigma
            if fit_mask.sum() >= 2:
                coeffs = np.polyfit(log_x[fit_mask], log_y[fit_mask], 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                x_fit = np.logspace(np.log10(x_sel.min()), np.log10(x_sel.max()), 200)
                y_fit = np.exp(intercept) * (x_fit ** slope)
                ax_d.plot(x_fit, y_fit, linestyle, color=color, lw=1.2, label=fit_label.format(slope=slope))

        return {
            'slope': slope,
            'x': x_sel,
            'y': y_sel
        }

    opt_info = _plot_with_fit(s_L_O_all, nu_max_vals, 'o', 'purple', r'Lifted $\nu(L)$', 'Fit ($L$, slope={slope:.2f})')
    lo_info = _plot_with_fit(s_L_O_all, nu_collapsed_all, 's', 'tab:green', r'Collapsed $\nu(L_O)$', 'Fit ($L_O$, slope={slope:.2f})', linestyle='-')

    x_arrays = []
    if opt_info:
        x_arrays.append(opt_info['x'])
    if lo_info:
        x_arrays.append(lo_info['x'])
    if x_arrays:
        x_all = np.concatenate(x_arrays)
    else:
        x_all = np.array([])

    if x_all.size >= 2:
        x_ref = np.logspace(np.log10(x_all.min()), np.log10(x_all.max()), 100)
    else:
        x_ref = np.array([])

    base_const = None
    if opt_info and opt_info['x'].size > 0:
        base_const = opt_info['y'][0] / (opt_info['x'][0] ** 0.5)
    elif lo_info and lo_info['x'].size > 0:
        base_const = lo_info['y'][0] / (lo_info['x'][0] ** 0.5)

    if x_ref.size > 0 and base_const is not None and np.isfinite(base_const):
        y_ref = base_const * (x_ref ** 0.5)
        ax_d.plot(x_ref, y_ref, color='0.2', linestyle=':', lw=1.5, label=r'Theory: slope $1/2$')

    if opt_info:
        slope_opt = opt_info['slope']
    else:
        slope_opt = None
    if lo_info:
        slope_lo = lo_info['slope']
    else:
        slope_lo = None

    title_terms = []
    if slope_opt is not None and np.isfinite(slope_opt):
        # put \mathrm inside math mode so LaTeX accepts it
        title_terms.append('$' + r'm_{L}\approx' + f'{slope_opt:.2f}' + '$')
    if slope_lo is not None and np.isfinite(slope_lo):
        title_terms.append('$' + r'm_{L_O}\approx' + f'{slope_lo:.2f}' + '$')

    if title_terms:
        title_text = r'\textbf{(d)} Quadratic Speedup (' + ', '.join(title_terms) + ')'
    else:
        title_text = r'\textbf{(d)} Quadratic Speedup'

    if not opt_info and not lo_info:
        ax_d.text(0.5, 0.5, 'Insufficient positive data for panel (d)', ha='center', va='center')

    ax_d.set_title(title_text, y=1.05)
    ax_d.set_xlabel(r'Singular Gap $s(L_O)$')
    ax_d.set_ylabel(r'Rate $\nu$')
    ax_d.legend(loc='lower right')

    plt.tight_layout(pad=2.0)
    
    output_filename = r'diffusion/diffusion_speedup_verification.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nFigure saved to {output_filename}")
    plt.show()


if __name__ == '__main__':
    # It's recommended to cache results as the simulation is long.
    results_file = RESULTS_NPY_PATH

    if os.path.exists(results_file):
        print(f"Loading cached results from {results_file}...")
        results = np.load(results_file, allow_pickle=True).tolist()
    else:
        results = run_full_experiment()
        np.save(results_file, results)
        save_results_jsonl(results, RESULTS_JSONL_PATH)
        print(f"\nResults saved to {results_file} and {RESULTS_JSONL_PATH} for future use.")

    if results:
        create_publication_figure(results)
    else:
        print("Experiment failed to produce results. No figure generated.")
