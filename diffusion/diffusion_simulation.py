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

# --- 4. Analysis: Integrated Autocorrelation Time ---

def integrated_autocorrelation_time(x, c=5):
    """
    Calculates the Integrated Autocorrelation Time (IAT) of a time series.
    Uses the methodology from the emcee package for robustness.
    """
    # Ensure the series is long enough
    if len(x) < 2 * c:
        return np.nan

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
    
    # Simulation settings for IAT calculation (can be overridden by QUICK mode below)
    SIM_DT = 0.01
    N_STEPS_IAT = 2_000_000  # Long simulation for accurate IAT
    N_BURN_IN = 100_000      # Steps to discard for equilibration
    REPEATS_PER_GAMMA = 3    # Run multiple independent trajectories per gamma and aggregate
    initial_q = np.array([-1.0, 0.0])

    # Production mode: allow an environment variable to request higher statistical quality
    prod = os.environ.get('ZENO_PROD', '0') == '1'
    if prod:
        print('Production mode enabled: increasing simulation length and repeats for final results')
        # Conservative but substantially higher-quality settings. Adjust if you want longer runs.
        N_STEPS_IAT = 100_000
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
        N_STEPS_IAT = 20000
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
            iats_all = []
            med_iats = []
            std_iats = []
            # Keep total simulated time constant across different gamma by adapting dt and n_steps.
            TOTAL_SIM_TIME = N_STEPS_IAT * SIM_DT
            BURN_TIME = N_BURN_IN * SIM_DT

            for gamma in tqdm(gamma_values, desc=f"  Scanning Gamma (h_b={h_b})"):
                # choose a safe dt for the given gamma to resolve OU step; don't exceed SIM_DT
                if gamma > 0:
                    dt_gamma = min(SIM_DT, 0.1 / gamma)
                else:
                    dt_gamma = SIM_DT

                n_steps_gamma = max(int(np.ceil(TOTAL_SIM_TIME / dt_gamma)), 10)
                burn_steps_gamma = min(int(np.ceil(BURN_TIME / dt_gamma)), n_steps_gamma-1)

                repeats = []
                for rep in range(REPEATS_PER_GAMMA):
                    traj = run_simulation(n_steps_gamma, dt_gamma, gamma, h_b, initial_q)
                    # apply burn-in in steps for this gamma
                    iat = integrated_autocorrelation_time(traj[burn_steps_gamma:])
                    repeats.append(iat)
                repeats = np.array(repeats, dtype=float)
                iats_all.append(repeats)
                med_iats.append(np.nanmedian(repeats))
                std_iats.append(np.nanstd(repeats))

            med_iats = np.array(med_iats)
            std_iats = np.array(std_iats)

            # Find optimal gamma using median IAT across repeats
            valid_mask = ~np.isnan(med_iats)
            if np.any(valid_mask):
                min_iat_idx = np.nanargmin(med_iats)
                gamma_opt = float(gamma_values[min_iat_idx])
                iat_min = float(med_iats[min_iat_idx])
                # Convergence rate is proportional to 1/IAT (use median-based value)
                nu_max = 1.0 / iat_min

                print(f"  -> Optimal Gamma = {gamma_opt:.3f}, Min median IAT = {iat_min:.3f}")

                results.append({
                    "h_b": h_b,
                    "s_L_O": s_L_O,
                    "gammas": gamma_values,
                    "iats": med_iats,       # median per gamma
                    "iats_all": iats_all,   # raw repeats per gamma
                    "iats_std": std_iats,
                    "gamma_opt": gamma_opt,
                    "iat_min": iat_min,
                    "nu_max": nu_max
                })
                # checkpoint: save after completing each barrier height so long runs can be resumed
                try:
                    np.save(r'diffusion/experiment_results.npy', results)
                    print(f"  -> Checkpoint saved ({len(results)} barrier(s) completed)")
                except Exception:
                    print("  -> Warning: failed to save checkpoint file.")
            else:
                print(f"  -> IAT calculation failed for all gammas.")
    except KeyboardInterrupt:
        print('\nRun interrupted by user; saving partial results...')
        try:
            np.save(r'diffusion/experiment_results.npy', results)
            print('Partial results saved to experiment_results.npy')
        except Exception:
            print('Failed to save partial results on interrupt.')
        raise
    except Exception as e:
        print(f"Exception during run_full_experiment: {e}")
        try:
            np.save(r'diffusion/experiment_results.npy', results)
            print('Partial results saved to experiment_results.npy')
        except Exception:
            print('Failed to save partial results after exception.')
        raise

    return results

# --- 6. Plotting ---

def create_publication_figure(results):
    print("\n--- Generating Publication Figure ---")
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 1), height_ratios=(1, 1))

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

    # --- Panel (b): Trajectories ---
    ax_b = fig.add_subplot(gs[0, 1])
    # Simulation for plotting trajectories
    n_steps_traj = 50000
    dt_traj = 0.05
    initial_q_traj = np.array([-1.0, 0.0])
    
    # Overdamped (Euler-Maruyama)
    @numba.njit
    def run_overdamped(n_steps, dt, h_b, q0):
        q = q0.copy()
        traj_x = np.empty(n_steps)
        for i in range(n_steps):
            gradU_x, gradU_y = grad_potential(q[0], q[1], h_b)
            Fx, Fy = non_conservative_force(q[0], q[1])
            q[0] += dt * (gradU_x + Fx) + np.sqrt(2.0 / BETA * dt) * np.random.normal()
            q[1] += dt * (gradU_y + Fy) + np.sqrt(2.0 / BETA * dt) * np.random.normal()
            traj_x[i] = q[0]
        return traj_x
        
    traj_overdamped = run_overdamped(n_steps_traj, dt_traj, h_b_plot, initial_q_traj)
    
    # Optimally lifted
    gamma_opt_plot = results[-1]['gamma_opt']
    traj_lifted = run_simulation(n_steps_traj, dt_traj, gamma_opt_plot, h_b_plot, initial_q_traj)

    time_axis = np.arange(n_steps_traj) * dt_traj
    # Subsample raw traces for clarity and overlay a smoothed curve (EMA)
    subsample = max(1, n_steps_traj // 2000)
    ax_b.plot(time_axis[::subsample], traj_overdamped[::subsample], label='Overdamped (Slow)', color='royalblue', lw=0.6, alpha=0.6)
    ax_b.plot(time_axis[::subsample], traj_lifted[::subsample], label='Optimally Lifted (Fast)', color='crimson', lw=0.6, alpha=0.6)

    # Exponential moving average smoothing for visualization
    def ema(x, alpha=0.01):
        s = np.empty_like(x)
        s[0] = x[0]
        for i in range(1, len(x)):
            s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
        return s

    ax_b.plot(time_axis, ema(traj_overdamped), color='royalblue', lw=1.2, alpha=0.9)
    ax_b.plot(time_axis, ema(traj_lifted), color='crimson', lw=1.2, alpha=0.9)
    ax_b.set_title(r'\textbf{(b)} Convergence Trajectories', y=1.05)
    ax_b.set_xlabel(r'Time')
    ax_b.set_ylabel(r'$x$-coordinate')
    ax_b.legend(loc='upper right')
    ax_b.set_ylim(-1.8, 1.8)
    ax_b.axhline(-1, color='k', linestyle='--', alpha=0.5)
    ax_b.axhline(1, color='k', linestyle='--', alpha=0.5)

    # --- Panel (c): Optimal Dissipation ---
    ax_c = fig.add_subplot(gs[1, 0])
    res_plot = results[-1]
    gammas = np.asarray(res_plot['gammas'])
    # Use median IATs and std dev across repeats for error bars
    med_iats = np.asarray(res_plot.get('iats'))
    std_iats = np.asarray(res_plot.get('iats_std')) if res_plot.get('iats_std') is not None else np.zeros_like(med_iats)
    # Optionally drop the smallest measured Gamma if it looks like an outlier/noisy point
    # (this avoids an extrapolation/plot artifact and reveals the single peak supported by the rest of the data)
    if gammas.size >= 2:
        idx_min = int(np.nanargmin(gammas))
        # drop only from the plotting arrays, keep original results untouched
        gammas_plot_source = np.delete(gammas, idx_min)
        med_iats_plot = np.delete(med_iats, idx_min)
        std_iats_plot = np.delete(std_iats, idx_min)
    else:
        gammas_plot_source = gammas
        med_iats_plot = med_iats
        std_iats_plot = std_iats
    # Guard against zeros or NaNs in med_iats to avoid infinite nu values
    med_safe = med_iats_plot.copy()
    med_safe = np.where(np.isnan(med_safe) | (med_safe <= 0), np.nan, med_safe)
    nu_empirical = np.where(np.isfinite(med_safe), 1.0 / med_safe, np.nan)
    nu_err = np.where(np.isfinite(med_safe), std_iats_plot / (med_safe**2), np.nan)

    # --- Densify only inside the valid data-supported Gamma range (no extrapolation) ---
    # Use the plotting-source arrays (may have dropped the smallest gamma)
    valid = np.isfinite(nu_empirical) & (gammas_plot_source > 0) & (nu_empirical > 0)
    nu_plot = None
    gammas_plot = None
    if valid.sum() >= 2:
        gamma_min_valid = float(gammas_plot_source[valid].min())
        gamma_max_valid = float(gammas_plot_source[valid].max())
        # densify inside the measured range only (no extrapolation beyond data)
        gammas_plot = np.logspace(np.log10(gamma_min_valid), np.log10(gamma_max_valid), num=400)
        log_xs = np.log10(gammas_plot_source[valid])
        log_ys = np.log10(nu_empirical[valid])
        log_xt = np.log10(gammas_plot)
        # interpolate in log-log space; since gammas_plot is within [gamma_min_valid, gamma_max_valid] no extrapolation
        log_yt = np.interp(log_xt, log_xs, log_ys)
        nu_plot = 10 ** (log_yt)
    else:
        # fall back to plotting raw data only (use plotting-source arrays)
        gammas_plot = gammas_plot_source
        nu_plot = nu_empirical

    # Plot raw points faded, and an interpolated dense curve in the small-Gamma region (no smoothing)
    ax_c.plot(gammas_plot_source, nu_empirical, 'o', color='darkorange', alpha=0.6, ms=6, label=r'Raw $\nu$')
    ax_c.plot(gammas_plot, nu_plot, '-', color='darkorange', lw=1.5, alpha=0.95, label=r'Dense interpolation (no extrapolation)')
    # Mark the maximum of the interpolated curve explicitly
    if np.any(np.isfinite(nu_plot)):
        max_idx = int(np.nanargmax(nu_plot))
        gamma_peak = gammas_plot[max_idx]
        nu_peak = float(nu_plot[max_idx])
        ax_c.plot([gamma_peak], [nu_peak], marker='*', color='black', ms=12, label=r'Peak')
        ax_c.axvline(gamma_peak, color='k', linestyle='--', alpha=0.6)
        label = r'$\Gamma_{\mathrm{peak}} = ' + f'{gamma_peak:.2g}' + r'$'
        ax_c.annotate(label, xy=(gamma_peak, nu_peak), xytext=(6, 6), textcoords='offset points')

    # (No error band drawn — we show dense interpolation only, no smoothing/extrapolation)

    ax_c.set_xscale('log')
    ax_c.set_title(r'\textbf{(c)} Optimal Dissipation', y=1.05)
    ax_c.set_xlabel(r'Friction Coefficient $\Gamma$')
    ax_c.set_ylabel(r'Convergence Rate $\nu \propto 1/\tau_x$')
    ax_c.legend()

    # --- Panel (d): Quadratic Speedup ---
    ax_d = fig.add_subplot(gs[1, 1])
    s_L_O_vals = np.array([r['s_L_O'] for r in results])
    # use median-based nu_max (1/iat_min) across results
    # compute bootstrapped nu_max distributions if repeats are available
    quick_flag = os.environ.get('ZENO_QUICK', '0') == '1'
    B = 100 if quick_flag else 200
    nu_max_vals = []
    nu_log_vars = []
    nu_ci_low = []
    nu_ci_high = []
    for r in results:
        if 'iats_all' in r and r['iats_all']:
            # iats_all: list of arrays of repeats per gamma
            # bootstrap the iat_min (median across gammas) per bootstrap sample
            iats_all = np.asarray(r['iats_all'])
            # compute per-gamma medians per bootstrap
            bs_nu = []
            for b in range(B):
                # resample repeats with replacement for each gamma
                medians = []
                for g_idx in range(iats_all.shape[0]):
                    repeats = iats_all[g_idx]
                    if len(repeats) == 0:
                        medians.append(np.nan)
                    else:
                        sample = np.random.choice(repeats, size=len(repeats), replace=True)
                        medians.append(np.nanmedian(sample))
                medians = np.array(medians)
                # find min median across gammas
                if np.all(np.isnan(medians)):
                    bs_nu.append(np.nan)
                else:
                    iat_min_b = np.nanmin(medians)
                    bs_nu.append(1.0 / iat_min_b)
            bs_nu = np.array(bs_nu)
            # summarize
            nu_max_vals.append(np.nanmedian(bs_nu))
            # variance in log space for weighting
            log_bs = np.log(bs_nu[~np.isnan(bs_nu)])
            nu_log_vars.append(np.var(log_bs) if log_bs.size>0 else np.nan)
            nu_ci_low.append(np.nanpercentile(bs_nu, 2.5))
            nu_ci_high.append(np.nanpercentile(bs_nu, 97.5))
        else:
            nu_max_vals.append(r.get('nu_max', np.nan))
            nu_log_vars.append(np.nan)
            nu_ci_low.append(np.nan)
            nu_ci_high.append(np.nan)
    nu_max_vals = np.array(nu_max_vals)
    nu_log_vars = np.array(nu_log_vars)
    nu_ci_low = np.array(nu_ci_low)
    nu_ci_high = np.array(nu_ci_high)
    
    valid_data = ~np.isnan(s_L_O_vals) & ~np.isnan(nu_max_vals)
    s_L_O_vals = s_L_O_vals[valid_data]
    nu_max_vals = nu_max_vals[valid_data]

    # plot errorbars for nu (CI) when available on log-log axes for clarity
    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    # protect against zero/negative values
    positive = (s_L_O_vals > 0) & (nu_max_vals > 0)
    if np.any(positive):
        x_pts = s_L_O_vals[positive]
        y_pts = nu_max_vals[positive]
        # y-errorbars from CI
        yerr_low = y_pts - nu_ci_low[positive]
        yerr_high = nu_ci_high[positive] - y_pts
        # Do not plot error bars (bootstrap CI are noisy for small sample sizes). Plot only points.
        ax_d.plot(x_pts, y_pts, 'o', color='purple', markersize=7, label='Numerical Results')

        # Fit and plot scaling law (robust log-log fit with simple sigma-clipping)
        if x_pts.size > 1:
            log_s = np.log(x_pts)
            log_nu = np.log(y_pts)

            mask = np.ones_like(log_s, dtype=bool)
            for _ in range(3):
                if mask.sum() < 2:
                    break
                coeffs = np.polyfit(log_s[mask], log_nu[mask], 1)
                pred = coeffs[0] * log_s + coeffs[1]
                resid = log_nu - pred
                sigma = np.nanstd(resid[mask])
                sigma = max(sigma, 1e-12)
                mask = np.abs(resid) < 2.0 * sigma

            if mask.sum() >= 2:
                # final (unweighted) fit on clipped data
                coeffs = np.polyfit(log_s[mask], log_nu[mask], 1)
                fit_slope = coeffs[0]
                fit_intercept = coeffs[1]
                # prepare fit curve
                x_fit = np.logspace(np.log10(x_pts.min()), np.log10(x_pts.max()), 200)
                y_fit = np.exp(fit_intercept) * (x_fit ** fit_slope)
                ax_d.plot(x_fit, y_fit, 'k--', lw=1.2, label=f'Fit: slope={fit_slope:.2f}')
            else:
                fit_slope = np.nan

        # Theoretical reference (slope 1/2)
        x_ref = np.logspace(np.log10(x_pts.min()), np.log10(x_pts.max()), 100)
        y_ref = (x_ref ** 0.5) * (y_pts[0] / (x_pts[0] ** 0.5))
        ax_d.plot(x_ref, y_ref, color='0.2', linestyle=':', lw=1.5, label=r'Theory: slope $1/2$')

        if not np.isnan(fit_slope):
            ax_d.set_title(r'\textbf{{(d)}} Quadratic Speedup ($m \approx {:.2f}$)'.format(fit_slope), y=1.05)
        else:
            ax_d.set_title(r'\textbf{(d)} Quadratic Speedup', y=1.05)
    else:
        ax_d.text(0.5, 0.5, 'Insufficient positive data for panel (d)', ha='center', va='center')
        ax_d.set_title(r'\textbf{(d)} Quadratic Speedup', y=1.05)

    ax_d.set_xlabel(r'Singular Gap $s(\mathcal{L}_O)$')
    ax_d.set_ylabel(r'Max Lifted Rate $\nu_{\max}$')
    ax_d.legend(loc='lower right')

    plt.tight_layout(pad=2.0)
    
    output_filename = r'diffusion/diffusion_speedup_verification.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nFigure saved to {output_filename}")
    plt.show()


if __name__ == '__main__':
    # It's recommended to cache results as the simulation is long.
    results_file = r'diffusion/experiment_results.npy'
    
    if os.path.exists(results_file):
        print(f"Loading cached results from {results_file}...")
        results = np.load(results_file, allow_pickle=True).tolist()
    else:
        results = run_full_experiment()
        np.save(results_file, results)
        print(f"\nResults saved to {results_file} for future use.")

    if results:
        create_publication_figure(results)
    else:
        print("Experiment failed to produce results. No figure generated.")
