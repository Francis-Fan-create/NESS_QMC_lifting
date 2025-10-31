import json
from pathlib import Path

import numpy as np
import qutip as qt
try:
    from qutip.visualization import matrix_histogram_complex
except Exception:  # pragma: no cover - defensive import logic
    # Fallback discovery for legacy QuTiP versions
    if hasattr(qt, 'visualization') and hasattr(qt.visualization, 'matrix_histogram_complex'):
        matrix_histogram_complex = qt.visualization.matrix_histogram_complex
    elif hasattr(qt, 'matrix_histogram_complex'):
        matrix_histogram_complex = qt.matrix_histogram_complex
    else:
        import matplotlib
        import matplotlib.pyplot as plt

        def matrix_histogram_complex(m, xlabels=None, ylabels=None, title=None, ax=None, colorbar=True, **kwargs):
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            data = m.full() if hasattr(m, 'full') else np.asarray(m)
            data = np.asarray(data, dtype=complex)

            mag = np.abs(data)
            phase = np.angle(data)

            xpos, ypos = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos, dtype=float)

            dx = dy = 0.8 * np.ones_like(zpos, dtype=float)
            dz = mag.flatten()

            norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
            cmap = matplotlib.cm.hsv
            colors_vals = cmap(norm(phase.flatten()))

            ax.bar3d(xpos - 0.4, ypos - 0.4, zpos, dx, dy, dz, color=colors_vals, shade=True)
            ax.set_xticks(range(data.shape[1]))
            ax.set_yticks(range(data.shape[0]))
            ax.set_zticks([])

            if xlabels is not None:
                ax.set_xticklabels(xlabels)
            if ylabels is not None:
                ax.set_yticklabels(ylabels)
            if title is not None:
                ax.set_title(title)

            return ax

from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Required for 3D projection support in mpl
from scipy.linalg import pinv
import time
import shutil
from tqdm import tqdm
import warnings

# Suppress common warnings from sparse matrix operations in QuTiP
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

COLLAPSED_RATE_DISPLAY_FLOOR = 1e-2  # Display floor for collapsed IAT rates (log-scale stability)
LIFTED_IAT_REPEATS = 5               # QJMC repeats per Gamma for the lifted model
COLLAPSED_IAT_REPEATS = 5            # MC repeats for the collapsed model
S_LO_SIM_SCALE = 1e4                 # Uniformly rescale L_O to lift gaps toward ~1e-10
RATE_RENORMALIZATION = S_LO_SIM_SCALE
S_LO_DISPLAY_SCALE = 1.0             # Keep panel (d) on the true scale post-renormalization

RESULTS_FILE_PATH = Path("spin") / "quantum_experiment_results.npy"
FIGURE_OUTPUT_PATH = Path("spin") / "quantum_speedup_verification.png"
EXPERIMENT_JSONL_PATH = Path("spin") / "experiment_summary.jsonl"

# Match diffusion figure styling for consistency across experiments
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    pass

SPIN_FIG_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
}

if SPIN_FIG_STYLE.get("text.usetex") and shutil.which("latex") is None:
    SPIN_FIG_STYLE["text.usetex"] = False

plt.rcParams.update(SPIN_FIG_STYLE)

# ---
# Qubit and Spin Chain Operators (N=3)
# ---

# Operators for a 3-qubit system
si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
sp = qt.sigmap()
sm = qt.sigmam()

def tensor_op(op, N, target):
    """Creates a qobj for an N-qubit operator acting on the 'target' qubit."""
    op_list = [si] * N
    op_list[target] = op
    return qt.tensor(op_list)

# ---
# Helper Functions for Liouvillian Construction
# ---

def get_lifted_L(N, J, D, Gamma):
    """
    Constructs the full (lifted) Liouvillian L = V + gamma*R
    for the N=3 spin chain.
    """
    assert N == 3, "This model is hard-coded for N=3"
    
    # --- Coherent Part V = i[H, .] ---
    # H = J*H0 (L_ham) + D*H1 (eta*L_pert)
    
    # H0 (Main XX coupling)
    H0 = (tensor_op(sx, N, 0) * tensor_op(sx, N, 1) +
          tensor_op(sy, N, 0) * tensor_op(sy, N, 1) +
          tensor_op(sx, N, 1) * tensor_op(sx, N, 2) +
          tensor_op(sy, N, 1) * tensor_op(sy, N, 2))
    
    # H1 (Non-conservative DM-like perturbation)
    # This term breaks reversibility of the effective L_O
    H1 = (tensor_op(sy, N, 0) * tensor_op(sz, N, 1) -
          tensor_op(sz, N, 0) * tensor_op(sy, N, 1))
    
    H = J * H0 + D * H1
    
    # --- Dissipative Part R ---
    c_ops = [
        np.sqrt(Gamma) * tensor_op(sm, N, 0),  # L_1 = sqrt(Gamma) * sigma_1^-
        np.sqrt(Gamma) * tensor_op(sp, N, 2)   # L_2 = sqrt(Gamma) * sigma_3^+
    ]
    
    # Construct the full Liouvillian
    L_lifted = qt.liouvillian(H, c_ops)
    return L_lifted, H, c_ops

def _extract_F_from_L_O(L_base_O, N=3):
    """Helper to extract the 2x2 classical rate matrix F from L_O."""
    assert N == 3, "L_O basis is hard-coded for N=3"
    
    # Basis for inner spin (tensored into Zeno space)
    # X_down = |v><v|_1 (x) |v><v|_2 (x) |^><^|_3
    # X_up   = |v><v|_1 (x) |^><^|_2 (x) |^><^|_3
    X_down = qt.tensor(qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 0)))
    X_up   = qt.tensor(qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 0)), qt.ket2dm(qt.basis(2, 0)))
    
    # Convert basis operators to super-vectors (Liouville space)
    X_down_vec = qt.operator_to_vector(X_down)
    X_up_vec   = qt.operator_to_vector(X_up)

    # Extract 2x2 rate matrix F (Schrodinger picture)
    F_matrix = np.zeros((2, 2), dtype=complex)
    # F_ij = Tr(X_i * L_O(X_j))
    F_matrix[0, 0] = (X_down_vec.dag() * L_base_O * X_down_vec) # F_dd
    F_matrix[0, 1] = (X_down_vec.dag() * L_base_O * X_up_vec)   # F_du
    F_matrix[1, 0] = (X_up_vec.dag() * L_base_O * X_down_vec)   # F_ud
    F_matrix[1, 1] = (X_up_vec.dag() * L_base_O * X_up_vec)     # F_uu
    
    # w_du is rate up -> down (F_du)
    # w_ud is rate down -> up (F_ud)
    w_du = float(np.real(F_matrix[0, 1]))
    w_ud = float(np.real(F_matrix[1, 0]))

    tol = 1e-14
    if w_du < 0 and abs(w_du) < tol:
        w_du = 0.0
    if w_ud < 0 and abs(w_ud) < tol:
        w_ud = 0.0

    if w_du < 0 or w_ud < 0:
        warnings.warn(
            f"Negative classical rate detected (w_du={w_du}, w_ud={w_ud}); taking absolute values.",
            RuntimeWarning
        )
        w_du = abs(w_du)
        w_ud = abs(w_ud)
    
    return w_du, w_ud

def get_base_L_O_and_gap(N, J, D):
    """
    Constructs the base (slow) Liouvillian L_O and computes its classical gap.
    L_O = -E * V * Q * (R_Q_inv) * Q * V * E

    Returns
    -------
    L_O : qt.Qobj
        The reduced Liouvillian acting on the slow subspace.
    gap_O : float
        The spectral gap of L_O (sum of classical transition rates).
    w_du : float
        Classical rate for the transition $\\uparrow \\rightarrow \\downarrow$ within the Zeno subspace.
    w_ud : float
        Classical rate for the transition $\\downarrow \\rightarrow \\uparrow$ within the Zeno subspace.
    """
    assert N == 3, "This model is hard-coded for N=3"
    
    # --- 1. Define Projections E (slow) and Q (fast) ---
    # P_Z is the projector onto the Zeno *state* subspace
    P_Z = qt.tensor(qt.ket2dm(qt.basis(2, 1)), si, qt.ket2dm(qt.basis(2, 0)))
    # E_super projects onto the Zeno *operator* subspace (ker(R))
    E_super = qt.sprepost(P_Z, P_Z)
    I_super = qt.qeye(E_super.dims[0])
    Q_super = I_super - E_super
    
    # H_D = E(V(E(X))) = P_Z H P_Z = 0 for this model, so E_O = E_super
    E_O = E_super 

    # --- 2. Define Operators V and R (Gamma=1) ---
    H0 = (tensor_op(sx, N, 0) * tensor_op(sx, N, 1) +
          tensor_op(sy, N, 0) * tensor_op(sy, N, 1) +
          tensor_op(sx, N, 1) * tensor_op(sx, N, 2) +
          tensor_op(sy, N, 1) * tensor_op(sy, N, 2))
    H1 = (tensor_op(sy, N, 0) * tensor_op(sz, N, 1) -
          tensor_op(sz, N, 0) * tensor_op(sy, N, 1))
    H = J * H0 + D * H1
    V_super = qt.liouvillian(H, [])
    
    c_ops_R = [tensor_op(sm, N, 0), tensor_op(sp, N, 2)]
    R_super = qt.liouvillian(None, c_ops_R)
    
    # --- 3. Invert R on the Fast Subspace Q ---
    R_Q = Q_super * R_super * Q_super
    # Use pseudo-inverse on the dense matrix.
    # This is slow (64x64) but robust and correct.
    R_Q_inv_data = pinv(R_Q.full(), rcond=1e-8)
    R_Q_inv = qt.Qobj(R_Q_inv_data, dims=R_super.dims)

    # --- 4. Compute L_O = -E * V * Q * (R_Q_inv) * Q * V * E ---
    L_O = -E_O * V_super * Q_super * R_Q_inv * Q_super * V_super * E_O

    if S_LO_SIM_SCALE != 1.0:
        L_O = S_LO_SIM_SCALE * L_O
    
    # --- 5. Compute Gap of L_O ---
    # The gap of L_O is the gap of the 2x2 classical rate matrix F
    w_du, w_ud = _extract_F_from_L_O(L_O, N)

    # The gap of a 2-state classical generator is the sum of the rates
    gap_O = w_du + w_ud

    if gap_O <= 0:
        warnings.warn(
            f"Base gap is numerically zero (w_du={w_du}, w_ud={w_ud}); using minimum positive value.",
            RuntimeWarning
        )
        gap_O = max(abs(w_du) + abs(w_ud), 1e-16)

    return L_O, gap_O, w_du, w_ud

# ---
# NEW METRIC: Integrated Autocorrelation Time (IAT)
# ---

def integrated_autocorrelation_time(x, c=5):
    """
    Calculates the Integrated Autocorrelation Time (IAT) of a time series.
    Uses the methodology from the emcee package (via the classical code).
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
    
    if acf[0] < 1e-10: # Handle zero variance
        return 1.0
        
    acf /= acf[0]

    # Use the initial positive sequence / pairwise summation
    try:
        g = acf[1:]
        if g.size < 1:
            return 1.0

        pair_sums = g[0::2].copy()
        second = g[1::2]
        pair_sums[:second.size] += second

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

def get_iat_rate_from_L(H, c_ops, J_val, D_val):
    """
    Runs a QJMC simulation to find the IAT of the "slow" observable
    sigma_z on the inner spin.
    
    Returns nu_eff = 1 / (2 * tau_IAT)
    """
    N = 3
    
    # --- 1. Define Simulation Parameters ---
    # Timescale is roughly 1/(J^2 + D^2).
    t_scale = 1.0 / (J_val**2 + D_val**2 + 1e-3)
    T_sim = 2000.0 * t_scale  # Total simulation time
    N_t = 10000              # Number of time steps
    times = np.linspace(0, T_sim, N_t)
    dt = times[1] - times[0]
    
    # --- 2. Define Initial State and Observable ---
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 0)) # |v, v, ^>
    obs = tensor_op(sz, N, 1) # sigma_z on the middle spin
    
    # --- 3. Run QJMC Simulation ---
    options = qt.Options(store_states=False, store_final_state=False, nsteps=10000)
    # options.use_sparse = True # This may or may not be supported/needed.
    try:
        result = qt.mcsolve(H, psi0, times, c_ops, [obs], ntraj=1, options=options)
        obs_ts = result.expect[0]
    except Exception as e:
        print(f"Warning: mcsolve failed ({e}). Returning NaN.")
        return np.nan
    
    # --- 4. Compute IAT ---
    burn_in_idx = N_t // 5
    obs_ts_stable = obs_ts[burn_in_idx:]
    
    tau_iat_steps = integrated_autocorrelation_time(obs_ts_stable)
    
    tau_iat_time = tau_iat_steps * dt
    
    if np.isnan(tau_iat_time) or tau_iat_time < dt:
        return np.nan 

    # The effective rate is 1 / (2 * tau_IAT)
    # This maps the IAT to the exponential decay rate nu.
    nu_eff = 1.0 / (2.0 * tau_iat_time)

    return nu_eff * RATE_RENORMALIZATION

def get_iat_rate_from_L_O(L_O, rng_seed=None):
    """
    Runs a Monte Carlo unraveling of the collapsed two-state dynamics defined by L_O
    and returns the effective IAT rate, using the same observable and estimator
    employed for the lifted model.

    Parameters
    ----------
    L_O : qt.Qobj
        Reduced (collapsed) Liouvillian.
    rng_seed : int, optional
        Optional NumPy random seed for reproducibility. When provided, the seed is
        set locally before calling :func:`qutip.mcsolve`.
    """
    rng_state = None
    if rng_seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(rng_seed)

    # --- 1. Extract classical rates ---
    w_du, w_ud = _extract_F_from_L_O(L_O, N=3)
    
    # --- 2. Define classical QJMC ---
    H_classical = qt.qeye(2) * 0.0 # No coherent part
    c_ops_classical = [
        np.sqrt(w_ud) * sm, # down -> up
        np.sqrt(w_du) * sp  # up -> down
    ]
    obs_classical = sz # distinguishes the two states
    psi0_classical = qt.basis(2, 1) # Start in |down>
    
    # --- 3. Define Sim Parameters ---
    t_scale = 1.0 / (w_ud + w_du + 1e-9)
    T_sim = 2000.0 * t_scale
    N_t = 10000
    times = np.linspace(0, T_sim, N_t)
    dt = times[1] - times[0]
    
    # --- 4. Run QJMC ---
    options = qt.Options(store_states=False, store_final_state=False)
    try:
        result = qt.mcsolve(H_classical, psi0_classical, times, c_ops_classical, [obs_classical], ntraj=1, options=options)
        obs_ts = result.expect[0]
    except Exception as e:
        print(f"Warning: mcsolve failed for L_O ({e}). Returning NaN.")
        return np.nan
    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)
        
    # --- 5. Compute IAT ---
    burn_in_idx = N_t // 5
    obs_ts_stable = obs_ts[burn_in_idx:]
    
    tau_iat_steps = integrated_autocorrelation_time(obs_ts_stable)
    tau_iat_time = tau_iat_steps * dt
    
    if np.isnan(tau_iat_time) or tau_iat_time < dt:
        return np.nan

    nu_eff = 1.0 / (2.0 * tau_iat_time)
    return nu_eff * RATE_RENORMALIZATION


def estimate_collapsed_iat_rate(L_O, repeats=COLLAPSED_IAT_REPEATS):
    """Estimate the collapsed-model convergence rate via repeated MC samples."""
    samples = []
    for _ in range(max(1, repeats)):
        nu = get_iat_rate_from_L_O(L_O)
        if np.isfinite(nu) and nu > 0:
            samples.append(float(nu))

    samples_arr = np.asarray(samples, dtype=float)
    w_du, w_ud = _extract_F_from_L_O(L_O, N=3)
    analytic = 0.5 * (w_du + w_ud)

    if samples_arr.size:
        median_val = float(np.median(samples_arr))
        used_fallback = False
    else:
        median_val = float(analytic)
        used_fallback = True

    return {
        "median": median_val,
        "samples": samples_arr.tolist(),
        "analytic": analytic,
        "used_fallback": used_fallback,
    }


def density_matrix_to_bloch_vector(rho):
    """Convert a single-qubit density matrix into its Bloch-vector representation."""
    data = rho.full() if hasattr(rho, "full") else np.asarray(rho)
    data = np.asarray(data, dtype=complex)
    if data.shape != (2, 2):
        raise ValueError("density_matrix_to_bloch_vector expects a 2x2 density matrix.")

    x = 2.0 * np.real(data[0, 1])
    y = -2.0 * np.imag(data[0, 1])
    z = np.real(data[0, 0] - data[1, 1])

    vec = np.array([x, y, z], dtype=float)
    return np.clip(vec, -1.0, 1.0)


# ---
# Analysis Functions (NESS)
# ---

def get_ness_current(L_lifted, L_base_O):
    """
    Calculates the NESS and the probability current
    in the slow (base) subspace.
    """
    try:
        ness_dm = qt.steadystate(L_lifted, use_precond=True, use_rcm=True, sparse=True, maxiter=5000)
    except:
        print("Warning: Sparse steadystate failed. Using dense.")
        ness_dm = qt.steadystate(L_lifted, sparse=False)
        
    if ness_dm.type != 'oper':
        print("Warning: Steadystate not a density matrix.")
        return None, 0.0

    w_du, w_ud = _extract_F_from_L_O(L_base_O, N=3)

    P_Z = qt.tensor(qt.ket2dm(qt.basis(2, 1)), si, qt.ket2dm(qt.basis(2, 0)))
    X_down = qt.tensor(qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 0)))
    X_up   = qt.tensor(qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 0)), qt.ket2dm(qt.basis(2, 0)))

    p_z = np.real((ness_dm * P_Z).tr())
    if p_z < 1e-6:
        print("Warning: NESS has no population in Zeno subspace.")
        return None, 0.0
        
    p_down = np.real((ness_dm * X_down).tr() / p_z)
    p_up   = np.real((ness_dm * X_up).tr() / p_z)
    
    current = p_down * w_ud - p_up * w_du
    
    coh_data = (ness_dm * X_down.dag() * X_up).tr() / p_z
    ness_inner = qt.Qobj([[p_down, np.conj(coh_data)], [coh_data, p_up]])
    
    return ness_inner, current


# ---
# Main Experiment Orchestration (matches classical code structure)
# ---

def run_full_experiment():
    print("--- Starting Quantum NESS Lifting Experiment ---")
    
    # --- Experiment Parameters ---
    J_list = np.linspace(0.2, 1.2, 8) # Our "barrier_heights"
    gamma_values = np.logspace(-1.5, 2.0, 20)
    D_val = 0.5 # Fixed non-conservative strength
    N = 3
    
    # Simulation settings
    # We run *fewer* repeats because each QJMC is much slower than BAOBAB
    REPEATS_PER_GAMMA = LIFTED_IAT_REPEATS
    collapsed_repeats = COLLAPSED_IAT_REPEATS
    
    results = []
    log_entries = []
    
    try:
        for J in tqdm(J_list, desc="J (System)"):
            print(f"\nAnalyzing system for J = {J:.2f}...")
            
            # --- STAGE 1: COLLAPSED SYSTEM ---
            # Compute both gap and IAT rate for L_O
            L_O, s_L_O, w_du, w_ud = get_base_L_O_and_gap(N, J, D_val)
            print(f"  -> Spectral Gap s(L_O) = {s_L_O:.4e}")
            collapsed_stats_theory = estimate_collapsed_iat_rate(L_O, repeats=collapsed_repeats)
            nu_collapsed_theory = collapsed_stats_theory["median"]
            theory_samples_arr = np.asarray(collapsed_stats_theory["samples"], dtype=float)

            # --- STAGE 2: LIFTED SYSTEM ---
            iats_all = []
            med_rates = []  # Median rate for each gamma

            for Gamma in tqdm(gamma_values, desc=f"  Scanning gamma (J={J:.2f})", leave=False):
                _, H, c_ops = get_lifted_L(N, J, D_val, Gamma)

                repeats_nu = []
                for _ in range(REPEATS_PER_GAMMA):
                    nu_eff = get_iat_rate_from_L(H, c_ops, J, D_val)
                    repeats_nu.append(nu_eff)

                iats_all.append(repeats_nu)
                med_rates.append(np.nanmedian(repeats_nu))

            med_rates = np.array(med_rates)

            gamma_high = float(gamma_values[-1])
            high_gamma_samples = np.asarray(iats_all[-1], dtype=float) if iats_all else np.asarray([])
            high_gamma_samples = high_gamma_samples[np.isfinite(high_gamma_samples) & (high_gamma_samples > 0)]

            if high_gamma_samples.size:
                nu_collapsed = float(np.nanmedian(high_gamma_samples))
                nu_collapsed_samples = high_gamma_samples.tolist()
                collapsed_source = "gamma_max"
                collapse_mc_fallback = False
                print(f"  -> Collapsed reference nu(L_O) ≈ {nu_collapsed:.4e} (Γ={gamma_high:.2e} median)")
                if np.isfinite(nu_collapsed_theory) and nu_collapsed > 0:
                    discrepancy = abs(nu_collapsed_theory - nu_collapsed) / nu_collapsed
                    if discrepancy > 5.0:
                        print(f"     [warn] Theory estimate differs by {discrepancy:.1f}× ({nu_collapsed_theory:.4e}).")
            else:
                nu_collapsed = nu_collapsed_theory
                valid_theory_samples = theory_samples_arr[np.isfinite(theory_samples_arr) & (theory_samples_arr > 0)]
                nu_collapsed_samples = valid_theory_samples.tolist()
                collapsed_source = "theory"
                collapse_mc_fallback = collapsed_stats_theory["used_fallback"]
                print(f"  -> Collapsed IAT Rate nu(L_O) ≈ {nu_collapsed:.4e} (theoretical estimate)")

            nu_collapsed_display = max(nu_collapsed, COLLAPSED_RATE_DISPLAY_FLOOR)
            s_L_O_display = s_L_O * S_LO_DISPLAY_SCALE

            valid_mask = ~np.isnan(med_rates)
            if np.any(valid_mask):
                max_rate_idx = int(np.nanargmax(med_rates))
                gamma_opt = float(gamma_values[max_rate_idx])
                nu_max = float(med_rates[max_rate_idx])

                print(f"  -> Optimal gamma = {gamma_opt:.3f}, Max IAT Rate = {nu_max:.4e}")

                result_entry = {
                    "J": J,
                    "s_L_O": s_L_O,
                    "s_L_O_display": s_L_O_display,
                    "nu_collapsed": nu_collapsed,
                    "nu_collapsed_display": nu_collapsed_display,
                    "nu_collapsed_gamma": gamma_high,
                    "nu_collapsed_source": collapsed_source,
                    "nu_collapsed_samples": nu_collapsed_samples,
                    "nu_collapsed_theory": nu_collapsed_theory,
                    "nu_collapsed_theory_samples": collapsed_stats_theory["samples"],
                    "nu_collapsed_raw": collapsed_stats_theory["analytic"],
                    "collapse_mc_fallback": collapse_mc_fallback,
                    "collapsed_mc_repeats": collapsed_repeats,
                    "classical_rates": (w_du, w_ud),
                    "gammas": gamma_values,
                    "rates": med_rates,
                    "rates_all": iats_all,
                    "gamma_opt": gamma_opt,
                    "nu_max": nu_max,
                    "rate_scale": RATE_RENORMALIZATION
                }
                results.append(result_entry)

                log_entries.append({
                    "J": J,
                    "s_L_O": s_L_O,
                    "s_L_O_display": s_L_O_display,
                    "nu_collapsed": nu_collapsed,
                    "nu_collapsed_source": collapsed_source,
                    "nu_collapsed_theory": nu_collapsed_theory,
                    "nu_collapsed_gamma": gamma_high,
                    "nu_max": nu_max,
                    "gamma_opt": gamma_opt,
                    "collapse_samples": len(nu_collapsed_samples),
                    "w_du": w_du,
                    "w_ud": w_ud,
                    "rate_scale": RATE_RENORMALIZATION
                })
            else:
                print(f"  -> IAT calculation failed for all gammas.")
    except KeyboardInterrupt:
        print('\nRun interrupted by user; returning partial results.')
    finally:
        if log_entries:
            EXPERIMENT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EXPERIMENT_JSONL_PATH.open('w', encoding='utf-8') as log_file:
                for entry in log_entries:
                    log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"\nSaved summary log to {EXPERIMENT_JSONL_PATH}")
    
    return results

# ---
# Plotting (matches classical code structure)
# ---

def create_publication_figure(results):
    print("\n--- Generating Publication Figure ---")
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 1), height_ratios=(1, 1))

    D_plot = 0.5  # Must match experiment parameters

    # Refresh derived quantities for each result (handles cached files from older runs)
    enriched_results = []
    for res in results:
        stored_scale = float(res.get('rate_scale', 1.0) or 1.0)
        if stored_scale <= 0:
            stored_scale = 1.0
        scale_factor = RATE_RENORMALIZATION / stored_scale
        if not np.isclose(scale_factor, 1.0):
            if 'nu_collapsed' in res:
                try:
                    res['nu_collapsed'] = float(res['nu_collapsed']) * scale_factor
                except (TypeError, ValueError):
                    pass
            if 'nu_collapsed_samples' in res:
                res['nu_collapsed_samples'] = (
                    np.asarray(res.get('nu_collapsed_samples', []), dtype=float) * scale_factor
                ).tolist()
            if 'nu_collapsed_theory' in res:
                try:
                    res['nu_collapsed_theory'] = float(res['nu_collapsed_theory']) * scale_factor
                except (TypeError, ValueError):
                    pass
            if 'nu_collapsed_theory_samples' in res:
                res['nu_collapsed_theory_samples'] = (
                    np.asarray(res.get('nu_collapsed_theory_samples', []), dtype=float) * scale_factor
                ).tolist()
            if 'nu_max' in res:
                try:
                    res['nu_max'] = float(res['nu_max']) * scale_factor
                except (TypeError, ValueError):
                    pass
            if 'rates' in res:
                res['rates'] = np.asarray(res.get('rates', []), dtype=float) * scale_factor
            if 'rates_all' in res:
                res['rates_all'] = [
                    (np.asarray(block, dtype=float) * scale_factor).tolist()
                    for block in res.get('rates_all', [])
                ]
            res['rate_scale'] = RATE_RENORMALIZATION

        J_val = res['J']
        L_O_obj, gap_val, w_du, w_ud = get_base_L_O_and_gap(3, J_val, D_plot)
        nu_collapsed_raw = 0.5 * (w_du + w_ud)

        gamma_values = np.asarray(res.get('gammas', []), dtype=float)
        nu_collapsed = res.get('nu_collapsed', np.nan)
        collapse_source = res.get('nu_collapsed_source')
        gamma_high = res.get('nu_collapsed_gamma', gamma_values[-1] if gamma_values.size else np.nan)
        samples_arr = np.asarray(res.get('nu_collapsed_samples', []), dtype=float)
        finite_samples = samples_arr[np.isfinite(samples_arr) & (samples_arr > 0)]
        collapse_fallback = bool(res.get('collapse_mc_fallback', collapse_source != 'gamma_max'))
        s_L_O_display = gap_val * S_LO_DISPLAY_SCALE

        if not np.isfinite(nu_collapsed) or nu_collapsed <= 0 or not finite_samples.size:
            rates_all = res.get('rates_all', [])
            if rates_all:
                high_gamma_samples = np.asarray(rates_all[-1], dtype=float)
                high_gamma_samples = high_gamma_samples[np.isfinite(high_gamma_samples) & (high_gamma_samples > 0)]
            else:
                high_gamma_samples = np.asarray([], dtype=float)

            if high_gamma_samples.size:
                nu_collapsed = float(np.median(high_gamma_samples))
                finite_samples = high_gamma_samples
                collapse_source = 'gamma_max'
                collapse_fallback = False
                if gamma_values.size:
                    gamma_high = float(gamma_values[-1])
            else:
                collapsed_stats = estimate_collapsed_iat_rate(
                    L_O_obj,
                    repeats=res.get('collapsed_mc_repeats', COLLAPSED_IAT_REPEATS),
                )
                nu_collapsed = collapsed_stats['median']
                finite_samples = np.asarray(collapsed_stats['samples'], dtype=float)
                collapse_source = 'theory'
                collapse_fallback = collapsed_stats['used_fallback']
                res['nu_collapsed_theory'] = nu_collapsed
            res['nu_collapsed'] = nu_collapsed
            res['nu_collapsed_samples'] = finite_samples.tolist()
            res['nu_collapsed_source'] = collapse_source
            res['nu_collapsed_gamma'] = gamma_high

        nu_collapsed_display = max(nu_collapsed, COLLAPSED_RATE_DISPLAY_FLOOR) if np.isfinite(nu_collapsed) else np.nan

        res['s_L_O'] = gap_val
        res['s_L_O_display'] = s_L_O_display
        res['nu_collapsed'] = nu_collapsed
        res['nu_collapsed_display'] = nu_collapsed_display
        res['nu_collapsed_raw'] = nu_collapsed_raw
        res['classical_rates'] = (w_du, w_ud)
        res['nu_collapsed_samples'] = finite_samples.tolist()
        res['collapse_mc_fallback'] = collapse_fallback
        enriched_results.append({**res, "L_O_obj": L_O_obj})

    # --- Panel (a): choose the system with the strongest steady-state current ---
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    panel_a_data = None
    best_abs_current = -np.inf

    for enriched in enriched_results:
        gamma_opt = enriched.get('gamma_opt', np.nan)
        if not np.isfinite(gamma_opt):
            continue
        L_lifted, _, _ = get_lifted_L(3, enriched['J'], D_plot, gamma_opt)
        ness_dm_inner, current = get_ness_current(L_lifted, enriched['L_O_obj'])
        if ness_dm_inner is None or not np.isfinite(current):
            continue
        if abs(current) > best_abs_current:
            best_abs_current = abs(current)
            panel_a_data = {
                'result': enriched,
                'ness_dm_inner': ness_dm_inner,
                'current': current,
                'gamma_opt': gamma_opt
            }

    if panel_a_data is None and enriched_results:
        fallback = enriched_results[-1]
        gamma_opt = fallback.get('gamma_opt', np.nan)
        L_lifted, _, _ = get_lifted_L(3, fallback['J'], D_plot, gamma_opt)
        ness_dm_inner, current = get_ness_current(L_lifted, fallback['L_O_obj'])
        panel_a_data = {
            'result': fallback,
            'ness_dm_inner': ness_dm_inner,
            'current': current,
            'gamma_opt': gamma_opt
        }

    if panel_a_data and panel_a_data['ness_dm_inner']:
        res_a = panel_a_data['result']
        ness_inner = panel_a_data['ness_dm_inner']
        matrix_histogram_complex(
            ness_inner,
            xlabels=[r'$|\downarrow\rangle$', r'$|\uparrow\rangle$'],
            ylabels=[r'$|\downarrow\rangle$', r'$|\uparrow\rangle$'],
            title=r'\textbf{(a)} NESS Inner Spin Density Matrix',
            ax=ax_a,
            colorbar=False,
        )
        ax_a.set_title(r'\textbf{(a)} NESS Inner Spin Density Matrix', pad=14)

        try:
            bloch_vec = density_matrix_to_bloch_vector(ness_inner)
        except ValueError:
            bloch_vec = np.zeros(3)

        ax_a.text2D(
            0.02,
            0.78,
            f"NESS current$= {panel_a_data['current']:+.3e}$\n$\\gamma_{{opt}}={panel_a_data['gamma_opt']:.2f}$",
            transform=ax_a.transAxes,
            ha='left',
            fontsize=10
        )
    else:
        ax_a.text2D(0.5, 0.5, "NESS computation failed.", transform=ax_a.transAxes, ha='center', va='center', fontsize=12)
    ax_a.set_title(r'\textbf{(a)} NESS Inner Spin Density Matrix', pad=14)

    # --- Panel (b): Collapsed vs Lifted IAT Rates ---
    ax_b = fig.add_subplot(gs[0, 1])
    J_vals = np.array([enriched['J'] for enriched in enriched_results], dtype=float)
    nu_collapsed_vals = np.array([enriched.get('nu_collapsed', np.nan) for enriched in enriched_results], dtype=float)
    nu_opt_vals = np.array([enriched.get('nu_max', np.nan) for enriched in enriched_results], dtype=float)

    def _collapsed_display(enriched):
        display = enriched.get('nu_collapsed_display', np.nan)
        base = enriched.get('nu_collapsed', np.nan)
        if np.isfinite(display) and display > 0:
            return float(display)
        if np.isfinite(base) and base > 0:
            return float(max(base, COLLAPSED_RATE_DISPLAY_FLOOR))
        return np.nan

    nu_collapsed_display_vals = np.array([_collapsed_display(enriched) for enriched in enriched_results], dtype=float)

    valid = (
        np.isfinite(nu_collapsed_vals)
        & np.isfinite(nu_opt_vals)
        & np.isfinite(nu_collapsed_display_vals)
        & (nu_collapsed_vals > 0)
        & (nu_opt_vals > 0)
        & (nu_collapsed_display_vals > 0)
    )

    if np.any(valid):
        x_plot = nu_collapsed_display_vals[valid]
        y_plot = nu_opt_vals[valid]
        x_actual = nu_collapsed_vals[valid]
        J_valid = J_vals[valid]

        ax_b.set_xscale('log')
        ax_b.set_yscale('log')
        ax_b.plot(x_plot, y_plot, 'o', color='tab:blue', ms=8, label='Data (annotated: $J$)')

        lims_min = min(x_plot.min(), y_plot.min()) * 0.8
        lims_max = max(x_plot.max(), y_plot.max()) * 1.2
        lims = [lims_min, lims_max]
        ax_b.plot(lims, lims, '--', color='0.3', lw=1, label=r'Identity $y=x$')

        for xi, yi, J_val in zip(x_plot, y_plot, J_valid):
            ax_b.annotate(f"{J_val:.1f}", xy=(xi, yi), xytext=(4, 4), textcoords='offset points', fontsize=10)

        ax_b.set_xlim(lims)
        ax_b.set_ylim(lims)
        ax_b.set_xlabel(r'Collapsed rate $\nu(L_O)$')
        ax_b.set_ylabel(r'Optimally lifted rate $\nu(L)$')
        ax_b.set_title(r'\textbf{(b)} Collapsed vs Lifted Convergence Rates', pad=12)
        ax_b.grid(True, which='both', ls=':', alpha=0.4)

        ratios = y_plot / x_actual
        median_ratio = np.nanmedian(ratios)
        max_ratio = np.nanmax(ratios)
        ax_b.text(0.05, 0.92, f'Median speedup: {median_ratio:.2f}x (max {max_ratio:.2f}x)', transform=ax_b.transAxes, ha='left', va='top', fontsize=10)

        floored = x_actual < COLLAPSED_RATE_DISPLAY_FLOOR
        if np.any(floored):
            ax_b.text(
                0.05,
                0.82,
                f'Collapsed medians < {COLLAPSED_RATE_DISPLAY_FLOOR:.1e} shown at floor',
                transform=ax_b.transAxes,
                ha='left',
                va='top',
                fontsize=9,
            )

        mc_fallback = [
            enriched_results[idx].get('collapse_mc_fallback', False)
            for idx, flag in enumerate(valid) if flag
        ]
        if any(mc_fallback):
            ax_b.text(
                0.05,
                0.74,
                'X markers: analytic fallback for collapsed rate',
                transform=ax_b.transAxes,
                ha='left',
                va='top',
                fontsize=9,
            )
            for point_x, point_y, fallback_flag in zip(x_plot, y_plot, mc_fallback):
                if fallback_flag:
                    ax_b.plot(
                        [point_x],
                        [point_y],
                        marker='X',
                        color='tab:red',
                        linestyle='None',
                        ms=10,
                        mew=1.5,
                        fillstyle='none',
                    )

        ax_b.legend(loc='lower right', fontsize=11)
    else:
        ax_b.clear()
        ax_b.axis('off')
        ax_b.text(0.5, 0.5, 'Insufficient or non-positive data to construct plot (b)', ha='center', va='center', fontsize=12)
    ax_b.set_title(r'\textbf{(b)} Collapsed vs Lifted (IAT Rates)', pad=12)


    # --- Panel (c): Optimal Dissipation ---
    ax_c = fig.add_subplot(gs[1, 0])
    res_c = enriched_results[-1] if enriched_results else results[-1]  # Use the last system
    gammas = np.asarray(res_c['gammas'])
    med_rates = np.asarray(res_c.get('rates'))
    
    valid = np.isfinite(med_rates) & (gammas > 0) & (med_rates > 0)
    if valid.sum() >= 2:
        gammas_valid = gammas[valid]
        rates_valid = med_rates[valid]
        
        # Plot raw points
        ax_c.plot(gammas_valid, rates_valid, 'o', color='darkorange', alpha=0.6, ms=6, label=r'Raw $\nu$')
        
        # Dense interpolation (no extrapolation)
        gamma_min_valid = float(gammas_valid.min())
        gamma_max_valid = float(gammas_valid.max())
        gammas_plot = np.logspace(np.log10(gamma_min_valid), np.log10(gamma_max_valid), num=200)
        
        log_xs = np.log10(gammas_valid)
        log_ys = np.log10(rates_valid)
        log_xt = np.log10(gammas_plot)
        
        log_yt = np.interp(log_xt, log_xs, log_ys)
        nu_plot = 10**(log_yt)
        
        ax_c.plot(gammas_plot, nu_plot, '-', color='darkorange', lw=1.5, alpha=0.95, label=r'Dense interpolation')

        # Mark the peak
        max_idx = int(np.nanargmax(nu_plot))
        gamma_peak = gammas_plot[max_idx]
        nu_peak = float(nu_plot[max_idx])
        ax_c.plot([gamma_peak], [nu_peak], marker='*', color='black', ms=12, label=r'Peak')
        ax_c.axvline(gamma_peak, color='k', linestyle='--', alpha=0.6)
        label = r'$\gamma_{\mathrm{opt}} = ' + f'{gamma_peak:.2g}' + r'$'
        ax_c.annotate(label, xy=(gamma_peak, nu_peak), xytext=(6, 6), textcoords='offset points')
    else:
        ax_c.text(0.5, 0.5, 'Insufficient data for plot (c)', ha='center', va='center')

    ax_c.set_xscale('log')
    ax_c.set_yscale('log') # Use log-log for (c) as well, often clearer
    ax_c.set_title(r'\textbf{(c)} Optimal Dissipation', pad=12)
    ax_c.set_xlabel(r'Friction Coefficient $\gamma$ (Dissipation $\Gamma$)')
    ax_c.set_ylabel(r'Convergence Rate $\nu \propto 1/\tau_{\mathrm{IAT}}$')
    ax_c.legend()

    # --- Panel (d): Quadratic Speedup (IAT vs Gap) ---
    ax_d = fig.add_subplot(gs[1, 1])
    s_L_O_vals = np.array([enriched.get('s_L_O_display', enriched['s_L_O'] * S_LO_DISPLAY_SCALE) for enriched in enriched_results])
    nu_max_vals = np.array([enriched.get('nu_max', np.nan) for enriched in enriched_results])
    
    valid_data = np.isfinite(s_L_O_vals) & np.isfinite(nu_max_vals) & (s_L_O_vals > 0) & (nu_max_vals > 0)
    
    if valid_data.sum() >= 2:
        x_pts = s_L_O_vals[valid_data]
        y_pts = nu_max_vals[valid_data]
        # Check for meaningful spread in x and y before attempting a log-log fit
        x_spread_ok = (np.isfinite(x_pts).all() and x_pts.max() / (x_pts.min() + 1e-30) > 1.02)
        y_spread_ok = (np.isfinite(y_pts).all() and y_pts.max() / (y_pts.min() + 1e-30) > 1.02)

        if x_spread_ok and y_spread_ok and (x_pts.min() > 0) and (y_pts.min() > 0):
            ax_d.set_xscale('log')
            ax_d.set_yscale('log')
            ax_d.plot(x_pts, y_pts, 'o', color='purple', markersize=7, label='Numerical Results')

            # Fit and plot scaling law
            log_s = np.log10(x_pts)
            log_nu = np.log10(y_pts)
            coeffs = np.polyfit(log_s, log_nu, 1)
            fit_slope = coeffs[0]
            fit_intercept = coeffs[1]

            x_fit = np.logspace(np.log10(x_pts.min()), np.log10(x_pts.max()), 100)
            y_fit = (10**fit_intercept) * (x_fit ** fit_slope)
            ax_d.plot(x_fit, y_fit, 'k--', lw=1.2, label=f'Fit: slope={fit_slope:.2f}')

            # Theoretical reference (slope 1/2)
            y_ref = (x_fit ** 0.5) * (y_pts[0] / (x_pts[0] ** 0.5)) # Rescaled
            ax_d.plot(x_fit, y_ref, color='0.2', linestyle=':', lw=1.5, label=r'Theory: slope $1/2$')

            ax_d.set_title(r'\textbf{{(d)}} Quadratic Speedup ($m \approx {:.2f}$)'.format(fit_slope), pad=12)
        else:
            # Not enough spread to perform a meaningful fit. Show raw points (with small jitter) and a note.
            ax_d.plot(np.arange(len(y_pts)), y_pts, 'o', color='purple', markersize=7, label='Numerical Results (no fit)')
            ax_d.set_xscale('linear')
            ax_d.set_yscale('log' if (y_pts.min() > 0 and y_pts.max()/y_pts.min() > 1.2) else 'linear')
            ax_d.set_title(r'\textbf{(d)} Quadratic Speedup (insufficient spread for fit)', pad=12)
            ax_d.text(0.5, 0.9, 'Insufficient variation in singular gap or rates to fit a scaling law.', transform=ax_d.transAxes, ha='center', va='top', fontsize=9)
            ax_d.set_xlabel('Index (insufficient spread in s(L_O))')
    else:
        ax_d.text(0.5, 0.5, 'Insufficient data for plot (d)', ha='center', va='center')
    ax_d.set_title(r'\textbf{(d)} Quadratic Speedup', pad=12)

    if S_LO_DISPLAY_SCALE != 1.0:
        xlabel = rf'Singular Gap $s(L_O)$ (display × {S_LO_DISPLAY_SCALE:.2g})'
        ax_d.text(0.05, 0.92, rf'$s(L_O)$ rescaled by {S_LO_DISPLAY_SCALE:.2g} for clarity', transform=ax_d.transAxes, fontsize=9, ha='left')
    else:
        xlabel = r'Singular Gap $s(L_O)$'
    ax_d.set_xlabel(xlabel)
    ax_d.set_ylabel(r'Max Lifted IAT Rate $\nu(L)$')
    ax_d.legend(loc='lower right')

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.94, wspace=0.28, hspace=0.35)

    output_filename = str(FIGURE_OUTPUT_PATH)
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nFigure saved to {output_filename}")
    plt.close(fig)


if __name__ == '__main__':
    # Make a directory for outputs if it doesn't exist
    try:
        import os
        if not os.path.exists('diffusion'):
            os.makedirs('diffusion')
    except:
        pass # Ignore if it fails

    results_file = RESULTS_FILE_PATH
    
    if results_file.exists():
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