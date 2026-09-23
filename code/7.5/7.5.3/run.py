# Section 7.5.3: index-1 Allen-Cahn saddle with a two-stage frozen metric.
# Save Figure 7, endpoint fields and repeated timings to outputs/7.5/7.5.3/.

import argparse as _t753_argparse
import csv as _t753_csv
import json as _t753_json
import os
import pickle as _t753_pickle
import statistics as _t753_statistics
import subprocess as _t753_subprocess
import sys as _t753_sys
import tempfile as _t753_tempfile
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye as speye, csc_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, spsolve_triangular
try:
    from sksparse.cholmod import cholesky as cholmod_cholesky
    USE_CHOLMOD = True
    print("Using CHOLMOD (scikit-sparse) for Cholesky decomposition")
except ImportError:
    USE_CHOLMOD = False
    print("CHOLMOD not available, using scipy's sparse triangular solver")
    from scipy.sparse.linalg import splu
    print("Warning: Using LU decomposition (install scikit-sparse for better performance)")


def build_laplacian_neumann(N):
    """Build 2D discrete Laplacian with Neumann BC on [0,1]^2, N x N grid."""
    h = 1.0 / (N - 1)
    e = np.ones(N)
    L1d = diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N), format='lil')
    L1d[0, 0] = 1.0; L1d[0, 1] = -1.0
    L1d[N-1, N-1] = 1.0; L1d[N-1, N-2] = -1.0
    L1d = L1d.tocsc() / h**2
    I_N = speye(N, format='csc')
    L2d = kron(I_N, L1d, format='csc') + kron(L1d, I_N, format='csc')
    return L2d, h


def grad_fn(u, L2d, eps):
    """Gradient: -Delta u + (u^3 - u)/eps^2"""
    return L2d @ u + (u**3 - u) / eps**2


def hess_matvec(u, v, L2d, eps):
    """Hessian-vector product: -Delta v + (3u^2 - 1)/eps^2 * v"""
    return L2d @ v + (3*u**2 - 1) / eps**2 * v


def cholesky_factor(M):
    """Factor the SPD metric and return a solver for M^{-1} b."""
    if USE_CHOLMOD:
        # Sparse Cholesky factorization with CHOLMOD.
        factor = cholmod_cholesky(M)
        return factor, lambda b: factor(b)
    else:
        # Sparse LU fallback.
        factor = splu(M)
        return factor, lambda b: factor.solve(b)


def run():
    # Grid, interface width, state/frame steps and residual stopping tolerance.
    N = 80
    n = N * N
    eps = 0.07
    dt_x_std = 1e-5
    dt_x_p = 0.2
    dt_v = 1e-2
    v_iter = 5
    max_iter_std = 2000000
    max_iter_p = 2000000
    tol = 1e-6

    L2d, h = build_laplacian_neumann(N)
    print(f"Allen-Cahn: N={N}, n={n}, eps={eps}, h={h:.5f}")

    # Share one perturbed planar-interface initial state between the two methods.
    np.random.seed(456)
    xx = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xx, xx)
    u0 = np.tanh((X.ravel() - 0.5) / (eps * np.sqrt(2))) + 0.05 * np.random.randn(n)

    timing_std = dict.fromkeys([
        'T_total', 'T_setup_update', 'T_apply_solve', 'T_eig', 'T_cert',
        'T_setup_phase1', 'T_setup_phase2', 'T_apply', 'T_solve',
        'T_eig_initial', 'T_eig_switch', 'T_eig_sigma', 'T_eig_update',
        'phase1_setup_count', 'phase2_setup_count', 'apply_count', 'solve_count',
        'state_updates', 'frame_update_count', 'initial_eig_fallback_count',
        'switch_eig_fallback_count', 'sigma_eig_fallback_count'], 0)
    timing_p = timing_std.copy()
    timing_std['factorization_backend'] = None
    timing_p['factorization_backend'] = 'CHOLMOD' if USE_CHOLMOD else 'SciPy SPLU fallback'
    t753_cert_spectra = {'standard': None, 'phisd': None}
    t753_cert_active = None
    print("Running Standard HiSD (dt_x=1e-5)...")
    u = u0.copy()

    # Initialize v (standard eigenproblem)
    t753_std_total = time.perf_counter()
    t753_std_initial = time.perf_counter()
    H_op = LinearOperator((n, n), matvec=lambda v: hess_matvec(u, v, L2d, eps))
    try:
        _, v_std = eigsh(H_op, k=1, which='SA')
        v_std = v_std[:, 0]
    except:
        timing_std['initial_eig_fallback_count'] += 1
        v_std = np.random.randn(n)
    v_std /= np.linalg.norm(v_std)
    timing_std['T_eig_initial'] = time.perf_counter() - t753_std_initial

    gnorms_std = []
    for it in range(max_iter_std):
        g = grad_fn(u, L2d, eps)
        gn = np.linalg.norm(g)
        gnorms_std.append(gn)
        if gn < tol:
            break

        # x-update: d = -g + 2v(v^T g)
        d = -g + 2.0 * v_std * (v_std @ g)
        u = u + dt_x_std * d
        timing_std['state_updates'] += 1

        # v-update: v = v - dt_v * (Hv - v(v^T Hv))
        t753_std_frame = time.perf_counter()
        for _ in range(v_iter):
            Hv = hess_matvec(u, v_std, L2d, eps)
            proj = Hv - v_std * (v_std @ Hv)
            v_std = v_std - dt_v * proj
            v_std /= np.linalg.norm(v_std)

        timing_std['T_eig_update'] += time.perf_counter() - t753_std_frame
        timing_std['frame_update_count'] += v_iter
        if (it + 1) % 200 == 0:
            print(f"  Std iter {it+1}: ||g||={gn:.4e}")

    print(f"  Standard: {len(gnorms_std)} iters, final ||g||={gnorms_std[-1]:.4e}")
    u_std_final = u.copy()
    timing_std['T_total'] = time.perf_counter() - t753_std_total

    print("Running Adaptive p-HiSD (dt_x=0.2)...")
    u = u0.copy()

    # First metric: M = -Delta_h + (2 / eps^2) I, fixed until the stagnation switch.
    print("  Constructing Phase 1 preconditioner (Cholesky)...")
    t753_p_total = time.perf_counter()
    t753_phase1 = time.perf_counter()
    timing_p['phase1_setup_count'] += 1
    mu_bj = 2.0 / eps**2
    M_frozen = L2d + mu_bj * speye(n, format='csc')
    M_frozen_csc = csc_matrix(M_frozen)
    M_frozen_factor, M_frozen_solve = cholesky_factor(M_frozen_csc)
    timing_p['T_setup_phase1'] = time.perf_counter() - t753_phase1

    t753_p_initial = time.perf_counter()
    H0 = L2d + diags((3 * u**2 - 1) / eps**2, 0, format='csc')
    try:
        _, v_p = eigsh(H0, k=1, M=M_frozen_csc, which='SA')
        v_p = v_p[:, 0]
    except:
        timing_p['initial_eig_fallback_count'] += 1
        v_p = np.random.randn(n)
    # M-normalization: v^T M v = 1
    t753_apply = time.perf_counter()
    Mv = M_frozen_csc @ v_p
    timing_p['T_apply'] += time.perf_counter() - t753_apply
    timing_p['apply_count'] += 1
    v_p = v_p / np.sqrt(v_p @ Mv)
    timing_p['T_eig_initial'] = time.perf_counter() - t753_p_initial

    gnorms_p = []
    switch_iter = None
    phase = 1
    M_solve = M_frozen_solve  # Reuse the selected sparse factorization.
    M_mat = M_frozen_csc

    for it in range(max_iter_p):
        g = grad_fn(u, L2d, eps)
        gn = np.linalg.norm(g)
        gnorms_p.append(gn)
        if gn < tol:
            break

        # Switch once if the last ten residuals decrease by less than 10%.
        if phase == 1 and it > 20:
            recent = gnorms_p[-10:]
            if len(recent) >= 10 and recent[-1] / (recent[0] + 1e-30) > 0.90:
                print(f"  Switching preconditioner at iteration {it} (||g||={gn:.4e})")
                print(f"  Constructing Phase 2 preconditioner (Cholesky)...")
                switch_iter = it
                phase = 2
                # Freeze a positively shifted Hessian at the switching state.
                t753_phase2 = time.perf_counter()
                timing_p['phase2_setup_count'] += 1
                H_sparse = L2d + diags((3*u**2 - 1) / eps**2, 0, format='csc')
                t753_sigma = time.perf_counter()
                try:
                    eig_min = eigsh(H_sparse, k=1, which='SA', return_eigenvectors=False)[0]
                except:
                    timing_p['sigma_eig_fallback_count'] += 1
                    eig_min = -100.0
                timing_p['T_eig_sigma'] += time.perf_counter() - t753_sigma
                sigma = max(0.0, -float(eig_min)) + 0.1
                H_shifted = H_sparse + sigma * speye(n, format='csc')
                H_shifted_csc = csc_matrix(H_shifted)
                M_chol_factor, M_chol_solve = cholesky_factor(H_shifted_csc)
                M_solve = M_chol_solve  # Reuse the second-stage factorization.
                M_mat = H_shifted_csc
                timing_p['T_setup_phase2'] += time.perf_counter() - t753_phase2
                t753_p_switch = time.perf_counter()
                try:
                    _, v_new = eigsh(H_sparse, k=1, M=M_mat, which='SA', v0=v_p)
                    v_p = v_new[:, 0]
                except Exception:
                    timing_p['switch_eig_fallback_count'] += 1
                    # If generalized eigensolve fails, keep current v_p but still M-normalize below.
                    pass
                t753_apply = time.perf_counter()
                Mv = M_mat @ v_p
                timing_p['T_apply'] += time.perf_counter() - t753_apply
                timing_p['apply_count'] += 1
                denom = float(v_p @ Mv)
                if denom <= 0 or not np.isfinite(denom):
                    raise RuntimeError("Phase 2 M-normalization failed: non-positive or non-finite v^T M v")
                v_p = v_p / np.sqrt(denom)
                timing_p['T_eig_switch'] += time.perf_counter() - t753_p_switch
                print(f"  ✓ Preconditioner FROZEN at iteration {it} (will NOT be updated)")
                print(f"  ✓ Using shifted Hessian: M = H(u_{it}) + {sigma:.4e}*I")

        # x-update: d = -M^{-1}g + 2v(v^T g)  [v is M-normalized]
        t753_solve = time.perf_counter()
        g_tilde = M_solve(g)
        timing_p['T_solve'] += time.perf_counter() - t753_solve
        timing_p['solve_count'] += 1
        d = -g_tilde + 2.0 * v_p * (v_p @ g)
        dt_x_eff = 1 if phase == 2 else dt_x_p
        u = u + dt_x_eff * d
        timing_p['state_updates'] += 1

        # v-update: v = v - dt_v * (M^{-1}Hv - v(v^T Hv))
        t753_p_frame = time.perf_counter()
        n_v_iter = v_iter
        for _ in range(n_v_iter):
            Hv = hess_matvec(u, v_p, L2d, eps)
            t753_solve = time.perf_counter()
            Minv_Hv = M_solve(Hv)
            timing_p['T_solve'] += time.perf_counter() - t753_solve
            timing_p['solve_count'] += 1
            proj = Minv_Hv - v_p * (v_p @ Hv)
            v_p = v_p - dt_v * proj

            # M-normalization: v^T M v = 1
            t753_apply = time.perf_counter()
            Mv = M_mat @ v_p
            timing_p['T_apply'] += time.perf_counter() - t753_apply
            timing_p['apply_count'] += 1
            denom = float(v_p @ Mv)
            if denom <= 0 or not np.isfinite(denom):
                raise RuntimeError("p-HiSD M-normalization failed: non-positive or non-finite v^T M v")
            v_p = v_p / np.sqrt(denom)

        timing_p['T_eig_update'] += time.perf_counter() - t753_p_frame
        timing_p['frame_update_count'] += n_v_iter
        if (it + 1) % 50 == 0:
            print(f"  p-HiSD iter {it+1}: ||g||={gn:.4e} (phase {phase})")

    print(f"  p-HiSD: {len(gnorms_p)} iters, final ||g||={gnorms_p[-1]:.4e}")
    u_p_final = u.copy()
    timing_p['T_total'] = time.perf_counter() - t753_p_total

    t753_cert = time.perf_counter()
    print("\n" + "="*60)
    print("VERIFICATION: Comparing the two saddle points")
    print("="*60)

    def energy(u, L2d, eps):
        grad_term = 0.5 * u @ (L2d @ u)
        potential_term = np.sum((u**2 - 1)**2) / (4 * eps**2)
        return grad_term + potential_term / (N * N)  # normalized by grid points

    E_std = energy(u_std_final, L2d, eps)
    E_p = energy(u_p_final, L2d, eps)

    print(f"\n1. Energy comparison:")
    print(f"   E(u_std)  = {E_std:.10e}")
    print(f"   E(u_p)    = {E_p:.10e}")
    print(f"   |E_std - E_p| = {abs(E_std - E_p):.10e}")

    diff = u_std_final - u_p_final
    l2_diff = np.linalg.norm(diff)
    linf_diff = np.max(np.abs(diff))
    rel_l2_diff = l2_diff / np.linalg.norm(u_std_final)

    print(f"\n2. Solution difference:")
    print(f"   ||u_std - u_p||_2     = {l2_diff:.10e}")
    print(f"   ||u_std - u_p||_inf   = {linf_diff:.10e}")
    print(f"   Relative L2 error     = {rel_l2_diff:.10e}")

    # Inspect the ordinary Hessian spectrum separately from residual-based stopping.
    print(f"\n3. Morse index verification:")
    H_std = L2d + diags((3*u_std_final**2 - 1) / eps**2, 0, format='csc')
    H_p = L2d + diags((3*u_p_final**2 - 1) / eps**2, 0, format='csc')

    try:
        t753_cert_active = ('standard', time.perf_counter())
        eigs_std = eigsh(H_std, k=5, which='SA', return_eigenvectors=False)
        t753_cert_spectra['standard'] = time.perf_counter() - t753_cert_active[1]
        t753_cert_active = None
        t753_cert_active = ('phisd', time.perf_counter())
        eigs_p = eigsh(H_p, k=5, which='SA', return_eigenvectors=False)
        t753_cert_spectra['phisd'] = time.perf_counter() - t753_cert_active[1]
        t753_cert_active = None

        n_neg_std = np.sum(eigs_std < 0)
        n_neg_p = np.sum(eigs_p < 0)

        print(f"   Standard HiSD: smallest 5 eigenvalues = {eigs_std}")
        print(f"   p-HiSD:        smallest 5 eigenvalues = {eigs_p}")
        print(f"   Standard HiSD: Morse index = {n_neg_std}")
        print(f"   p-HiSD:        Morse index = {n_neg_p}")
    except Exception as e:
        if t753_cert_active is not None:
            t753_cert_spectra[t753_cert_active[0]] = time.perf_counter() - t753_cert_active[1]
            t753_cert_active = None
        print(f"   Warning: Could not compute eigenvalues: {e}")

    print(f"\n4. Conclusion:")
    if rel_l2_diff < 1e-3 and abs(E_std - E_p) < 1e-6:
        print(f"   ✓ Both methods converged to the SAME saddle point!")
        print(f"   ✓ Relative difference: {rel_l2_diff:.2e} < 0.1%")
    else:
        print(f"   ✗ Methods may have converged to DIFFERENT saddle points.")
        print(f"   ✗ Relative difference: {rel_l2_diff:.2e}")

    print("="*60 + "\n")

    # Read-only endpoint diagnostics: history records PRE-update residuals.
    endpoint_residual_std = float(np.linalg.norm(grad_fn(u_std_final, L2d, eps)))
    endpoint_residual_p = float(np.linalg.norm(grad_fn(u_p_final, L2d, eps)))
    t753_cert_elapsed = time.perf_counter() - t753_cert
    for t753_record in (timing_std, timing_p):
        t753_record['T_cert'] = t753_cert_elapsed
        t753_record['T_cert_standard_spectrum'] = t753_cert_spectra['standard']
        t753_record['T_cert_phisd_spectrum'] = t753_cert_spectra['phisd']
        t753_record['T_setup_update'] = t753_record['T_setup_phase1'] + t753_record['T_setup_phase2']
        t753_record['T_apply_solve'] = t753_record['T_apply'] + t753_record['T_solve']
        t753_record['T_eig'] = sum(t753_record[k] for k in (
            'T_eig_initial', 'T_eig_switch', 'T_eig_sigma', 'T_eig_update'))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(
        script_dir, "..", "..", "..", "outputs", "7.5", "7.5.3"
    ))
    os.makedirs(output_dir, exist_ok=True)

    if globals().get('_TIMING753_SKIP_OUTPUTS', False):
        globals()['TIMING753_RESULT'] = locals().copy()
        return

    npz_path = os.path.join(output_dir, "saddle_points.npz")
    np.savez(npz_path,
             u_std=u_std_final.reshape(N, N),
             u_p=u_p_final.reshape(N, N),
             diff=diff.reshape(N, N),
             X=X, Y=Y,
             gnorms_std=np.asarray(gnorms_std),
             gnorms_p=np.asarray(gnorms_p))
    print(f"Solutions saved to: {npz_path}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    # Limit the displayed HiSD curve while retaining its full history in the data files.
    gnorms_std_plot = gnorms_std[:800]
    iters_std = np.arange(len(gnorms_std_plot))
    iters_p = np.arange(len(gnorms_p))

    ax.semilogy(iters_std, gnorms_std_plot, color='#1f77b4', linewidth=1.5,
                label=f'HiSD ', zorder=2)
    ax.semilogy(iters_p, gnorms_p, color='#d62728', linewidth=1.5,
                label=f'p-HiSD ', zorder=3)

    if switch_iter is not None:
        ax.axvline(x=switch_iter, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(switch_iter + 2, gnorms_p[switch_iter] * 2,
                f'Switch at iter {switch_iter}', fontsize=15, color='black',bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax.set_xlabel('Iteration $m$', fontsize=17)
    ax.set_ylabel('$\\|\\nabla E(u_m)\\|$', fontsize=17)
    ax.set_title('Allen-Cahn Equation ($\\xi=0.07$, $N=80$)', fontsize=17)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(output_dir, "5.3.pdf")

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {pdf_path}")
    global TIMING753_RESULT
    TIMING753_RESULT = locals().copy()


_T753_PARAMETERS = ('N', 'n', 'eps', 'dt_x_std', 'dt_x_p', 'dt_v', 'v_iter',
                    'max_iter_std', 'max_iter_p', 'tol', 'h', 'mu_bj')
_T753_CONTINUOUS = ('gnorms_std', 'gnorms_p', 'u_std_final', 'u_p_final', 'v_std', 'v_p',
                    'sigma', 'eig_min', 'E_std', 'E_p', 'diff', 'l2_diff', 'linf_diff',
                    'rel_l2_diff', 'eigs_std', 'eigs_p', 'endpoint_residual_std', 'endpoint_residual_p')
_T753_EXACT = ('u0', 'xx', 'X', 'Y', 'switch_iter', 'phase', 'n_neg_std', 'n_neg_p')


def _timing753_plain(value):
    if isinstance(value, dict):
        return {key: _timing753_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_timing753_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _timing753_plain(value.tolist())
    if isinstance(value, np.generic):
        return _timing753_plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _timing753_output_dir():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', '..', '..', 'outputs', '7.5', '7.5.3'))


def _timing753_capture(repetition):
    import scipy
    s = TIMING753_RESULT
    parameters = {key: s[key] for key in _T753_PARAMETERS}
    numerical = {key: s[key] for key in _T753_EXACT + _T753_CONTINUOUS if key in s}
    numerical['rng_state'] = np.random.get_state()
    numerical['USE_CHOLMOD'] = USE_CHOLMOD
    arrays = dict(u_std=s['u_std_final'].reshape(s['N'], s['N']),
                  u_p=s['u_p_final'].reshape(s['N'], s['N']), diff=s['diff'].reshape(s['N'], s['N']),
                  X=s['X'], Y=s['Y'], gnorms_std=np.asarray(s['gnorms_std']), gnorms_p=np.asarray(s['gnorms_p']))
    if repetition == 1:
        with np.load(s['npz_path'], allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays), 'Original NPZ fields changed.'
            for key in arrays:
                assert np.array_equal(saved[key], arrays[key], equal_nan=True), 'Retained NPZ data mismatch: ' + key
    numerical['npz'] = arrays
    numerical['plot_data'] = dict(standard_y=np.asarray(s['gnorms_std'][:800]),
        standard_x=np.arange(len(s['gnorms_std'][:800])), phisd_y=np.asarray(s['gnorms_p']),
        phisd_x=np.arange(len(s['gnorms_p'])), switch_iter=s['switch_iter'])
    methods = []
    for suffix, name in (('std', 'Standard HiSD'), ('p', 'two-stage p-HiSD')):
        history = s['gnorms_' + suffix]
        converged = bool(history[-1] < s['tol'])
        timing = s['timing_' + suffix].copy()
        timing['T_total_includes_certification'] = False
        timing['component_counters_are_inclusive'] = True
        record = dict(method=name, status='converged' if converged else 'max_iter',
            iterations=len(history), final_residual=float(history[-1]),
            endpoint_residual=s['endpoint_residual_' + suffix],
            morse_index=s.get('n_neg_' + suffix), smallest_five_eigenvalues=s.get('eigs_' + suffix),
            switch_iteration=s['switch_iter'] if suffix == 'p' else None,
            switch_residual=float(history[s['switch_iter']]) if suffix == 'p' and s['switch_iter'] is not None else None,
            sigma=s.get('sigma') if suffix == 'p' else None,
            eig_min=s.get('eig_min') if suffix == 'p' else None,
            final_phase=s['phase'] if suffix == 'p' else None,
            timing=timing)
        _timing753_verify(record, s['v_iter'])
        methods.append(record)
    assert methods[0]['timing']['T_cert'] == methods[1]['timing']['T_cert']
    record = dict(repetition=repetition, parameters=parameters, initial_numpy_seed=456, methods=methods,
        posthoc={key: s[key] for key in ('E_std', 'E_p', 'l2_diff', 'linf_diff', 'rel_l2_diff')},
        output_dir=s['output_dir'], process_id=os.getpid(),
        environment=dict(python=_t753_sys.version, executable=_t753_sys.executable,
            numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
            factorization_backend=methods[1]['timing']['factorization_backend'],
            thread_environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')}))
    return dict(numerics=numerical, record=record)


def _timing753_verify(record, v_iter):
    timing = record['timing']
    for key, value in timing.items():
        if key.startswith('T_') and key != 'T_total_includes_certification':
            if value is None and key in ('T_cert_standard_spectrum', 'T_cert_phisd_spectrum'):
                continue
            assert isinstance(value, (int, float)) and np.isfinite(value) and value >= 0, 'Invalid timing: ' + key
    assert timing['T_setup_update'] == timing['T_setup_phase1'] + timing['T_setup_phase2']
    assert timing['T_apply_solve'] == timing['T_apply'] + timing['T_solve']
    assert timing['T_eig'] == sum(timing[key] for key in
        ('T_eig_initial', 'T_eig_switch', 'T_eig_sigma', 'T_eig_update'))
    updates = record['iterations'] - int(record['status'] == 'converged')
    assert timing['state_updates'] == updates and timing['frame_update_count'] == v_iter * updates
    if record['method'] == 'Standard HiSD':
        assert timing['T_setup_update'] == timing['T_apply_solve'] == 0
        assert timing['phase1_setup_count'] == timing['phase2_setup_count'] == 0
        assert timing['apply_count'] == timing['solve_count'] == 0
    else:
        switched = int(record['switch_iteration'] is not None)
        assert timing['phase1_setup_count'] == 1 and timing['phase2_setup_count'] == switched
        assert timing['apply_count'] == 1 + switched + v_iter * updates
        assert timing['solve_count'] == (1 + v_iter) * updates


def _timing753_compare_native(reference, candidate):
    old, new = reference['numerics'], candidate['numerics']
    result = dict(passed=True, rtol=1e-10, atol=1e-10, field_differences={}, failures=[],
        comparison='Native independent eigsh streams; exact protocol/discrete results/counters; frame signs aligned only for comparison.')
    def exact(a, b, name):
        equal = np.array_equal(a, b, equal_nan=True) if isinstance(a, np.ndarray) else a == b
        if not equal:
            result['failures'].append(name + ': exact value differs')
    for key in _T753_PARAMETERS:
        exact(reference['record']['parameters'][key], candidate['record']['parameters'][key], key)
    for key in _T753_EXACT + ('USE_CHOLMOD',):
        if (key in old) != (key in new):
            result['failures'].append(key + ': availability differs')
        elif key in old:
            exact(old[key], new[key], key)
    for index, (a, b) in enumerate(zip(old['rng_state'], new['rng_state'])):
        exact(a, b, 'legacy_rng_' + str(index))
    def close(a, b, name, frame=False):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            result['failures'].append(name + ': shape differs')
            return
        raw_delta = float(np.max(np.abs(a-b))) if a.size else 0.0
        sign = -1 if frame and np.dot(a.ravel(), b.ravel()) < 0 else 1
        b = sign * b
        delta = float(np.max(np.abs(a-b))) if a.size else 0.0
        passed = bool(np.allclose(a, b, rtol=result['rtol'], atol=result['atol'], equal_nan=False))
        result['field_differences'][name] = dict(max_abs_difference=delta,
            raw_max_abs_difference=raw_delta, exact=bool(np.array_equal(a,b)), frame_sign=sign, passed=passed)
        if not passed:
            result['failures'].append(name + ': native reproducibility bound exceeded')
    for key in _T753_CONTINUOUS:
        if (key in old) != (key in new):
            result['failures'].append(key + ': availability differs')
        elif key in old:
            close(old[key], new[key], key, frame=key in ('v_std', 'v_p'))
    for key in old['npz']:
        close(old['npz'][key], new['npz'][key], 'npz.' + key)
    for key in ('standard_x', 'standard_y', 'phisd_x', 'phisd_y'):
        close(old['plot_data'][key], new['plot_data'][key], 'plot.' + key)
    for a, b in zip(reference['record']['methods'], candidate['record']['methods']):
        for key in ('method', 'status', 'iterations', 'morse_index', 'switch_iteration', 'final_phase'):
            exact(a[key], b[key], a['method'] + '.' + key)
        for key in a['timing']:
            if not key.startswith('T_'):
                exact(a['timing'][key], b['timing'][key], a['method'] + '.' + key)
    result['passed'] = not result['failures']
    return result


def _timing753_definitions():
    return dict(
        T_total='Original printing-inclusive solver wall-clock: Standard starts before H_op/initial eigsh and ends after u_std_final copy; p-HiSD starts before Phase-1 metric construction and ends after u_p_final copy. Excludes startup, common construction/initialization, shared posthoc certification, plotting and file I/O; includes instrumentation overhead.',
        T_setup_update='T_setup_phase1 + T_setup_phase2: original Phase-1 mu_bj/metric/CSC/factorization and actual Phase-2 Hessian/eig_min/sigma/shifted-metric/CSC/factorization. Standard is zero.',
        T_apply_solve='T_apply + T_solve: original explicit metric matrix-vector applications and factor solves, including nested frame operations. Standard is zero.',
        T_eig='T_eig_initial + T_eig_switch + T_eig_sigma + T_eig_update, including original frame HVP/projection/normalization and nested metric operations.',
        T_cert='One shared original posthoc block: energy/differences/endpoint Hessians/five-eigenvalue calculations/Morse counting/comparison and endpoint residual diagnostics. The same shared time is copied to both method rows and lies outside both T_total values; do not add the two T_cert rows.',
        accounting='Component counters are inclusive diagnostics and must not be summed to obtain T_total. Phase-2 sigma spectral work overlaps setup/update and eig; frame metric applications/solves overlap eig. Internal generalized eigsh metric work remains within eig. T_cert is one shared posthoc block repeated in both method rows; do not add its two rows.',
        iterations='Recorded residual points; state_updates counts executed steps. Last historical residual may precede the final endpoint when the budget is exhausted.',
        switch_iteration='Original zero-based loop index; Phase 2 stays frozen after the actual switch.')


def _publication_data(value):
    import re
    if isinstance(value, dict):
        return {key: _publication_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_publication_data(item) for item in value]
    if isinstance(value, str):
        def portable(match):
            path = match.group(0).replace('\\', '/')
            for marker in ('/outputs/', '/code/'):
                if marker in path:
                    return marker[1:] + path.rsplit(marker, 1)[1]
            return path.rstrip('/').rsplit('/', 1)[-1]
        value = re.sub(r"(?:/(?:Users|home|bicmr|private|tmp|opt|Applications)/|[A-Za-z]:[\\/](?:Users|Program Files)[\\/])[^\s\"'`<>;,]+", portable, value)
        return value
    return value


def _timing753_save(output_dir, artifacts, checks):
    records = [_timing753_plain(a['record']) for a in artifacts]
    metadata = dict(section='7.5.3', timing_repeats=len(records), unit='seconds',
        retained_original_repetition=1, T_total_includes_certification=False,
        component_counters_are_inclusive=True, definitions=_timing753_definitions(),
        run_policy='Three fresh sequential Python processes; original run(), method order, native eigsh RNG, parameters and backend; no warm-up or tuning. Only repetition 1 writes the original NPZ/PDF.',
        unsuccessful_run_policy='For unsuccessful methods, T_total is consumed wall-clock until termination, not time-to-solution.',
        native_reproducibility_checks=checks)
    raw = dict(metadata, repetitions=records)
    summary = dict(metadata, methods=[])
    csv_rows = []
    for method in range(2):
        first = records[0]['methods'][method]
        entry = dict(method=first['method'], status=first['status'], iterations=first['iterations'], timing_statistics={})
        for metric in first['timing']:
            if not metric.startswith('T_') or metric == 'T_total_includes_certification':
                continue
            values = [r['methods'][method]['timing'][metric] for r in records]
            stats = dict(raw=values, median=None, min=None, max=None)
            if all(value is not None for value in values):
                stats.update(median=_t753_statistics.median(values), min=min(values), max=max(values))
            entry['timing_statistics'][metric] = stats
            csv_rows.append(dict(method=first['method'], status=first['status'], iterations=first['iterations'],
                metric=metric, unit='seconds', repetition_1=values[0], repetition_2=values[1], repetition_3=values[2],
                median=stats['median'], min=stats['min'], max=stats['max'],
                T_total_includes_certification=False, component_counters_are_inclusive=True,
                accounting_note=metadata['definitions']['accounting']))
        summary['methods'].append(entry)
    arrays = {}
    for repeat, artifact in enumerate(artifacts, 1):
        for key, value in artifact['numerics']['npz'].items():
            arrays[f'repetition_{repeat}_{key}'] = value
        for key in ('u0', 'xx', 'v_std', 'v_p', 'eigs_std', 'eigs_p'):
            if key in artifact['numerics']:
                arrays[f'repetition_{repeat}_{key}'] = artifact['numerics'][key]
        rng_state = artifact['numerics']['rng_state']
        for key, value in zip(('name', 'keys', 'position', 'has_gauss', 'cached_gaussian'), rng_state):
            arrays[f'repetition_{repeat}_rng_{key}'] = np.asarray(value)
    np.savez_compressed(os.path.join(output_dir, 'timing753_raw.npz'), **arrays)
    for filename, document in (('timing753_raw.json', raw), ('timing753_summary.json', summary)):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as stream:
            _t753_json.dump(_publication_data(document), stream, indent=2, allow_nan=False)
            stream.write('\n')
    with open(os.path.join(output_dir, 'timing753_summary.csv'), 'w', newline='', encoding='utf-8') as stream:
        writer = _t753_csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    for method in summary['methods']:
        stats = method['timing_statistics']['T_total']
        print(f"{method['method']}: T_total median={stats['median']:.9g} s [{stats['min']:.9g}, {stats['max']:.9g}]")
    print(metadata['definitions']['accounting'])
    print('Timing files saved to:', output_dir)


# Run fresh repetitions, compare numerical outputs and retain the first plotted result.
def _timing753_dispatch():
    parser = _t753_argparse.ArgumentParser(description='Section 7.5.3: original experiment and three fresh-process timings.')
    parser.add_argument('--timing-child', type=int, choices=(1, 2, 3), help=_t753_argparse.SUPPRESS)
    parser.add_argument('--timing-artifact', help=_t753_argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timing_child is not None:
        if not args.timing_artifact:
            parser.error('--timing-child requires --timing-artifact')
        globals()['_TIMING753_SKIP_OUTPUTS'] = args.timing_child != 1
        run()
        with open(args.timing_artifact, 'wb') as stream:
            _t753_pickle.dump(_timing753_capture(args.timing_child), stream, protocol=5)
        return True
    timing_repeats = 3
    output_dir = _timing753_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    artifacts, checks = [], []
    source_bytes = open(__file__, 'rb').read()
    with _t753_tempfile.TemporaryDirectory(prefix='.timing753_', dir=output_dir) as scratch:
        for repeat in range(1, timing_repeats + 1):
            print(f'Formal repetition {repeat}/{timing_repeats}: fresh Python process', flush=True)
            destination = os.path.join(scratch, f'repetition_{repeat}.pickle')
            _t753_subprocess.run([_t753_sys.executable, os.path.abspath(__file__), '--timing-child', str(repeat),
                                 '--timing-artifact', destination], check=True)
            with open(destination, 'rb') as stream:
                artifact = _t753_pickle.load(stream)
            assert artifact['record']['output_dir'] == output_dir, 'Timing output directory mismatch.'
            check = _timing753_compare_native(artifacts[0] if artifacts else artifact, artifact)
            checks.append(dict(repetition=repeat, **check))
            if not check['passed']:
                with open(os.path.join(output_dir, 'timing753_failed_consistency.json'), 'w', encoding='utf-8') as stream:
                    _t753_json.dump(_publication_data(_timing753_plain(checks)), stream, indent=2, allow_nan=False)
                raise RuntimeError('STOP: native numerical consistency failed; no formal timing summary generated. ' + '; '.join(check['failures']))
            if open(__file__, 'rb').read() != source_bytes:
                raise RuntimeError('Source changed during timing.')
            artifacts.append(artifact)
        assert len({a['record']['process_id'] for a in artifacts}) == timing_repeats
        _timing753_save(output_dir, artifacts, checks)
    return True

if __name__ == '__main__':
    if _timing753_dispatch():
        raise SystemExit(0)
    run()
