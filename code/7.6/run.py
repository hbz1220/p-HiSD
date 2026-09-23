# Section 7.6: index-1 saddles of the reduced non-convex optimal-control objective.
# Compare HiSD with three fixed metrics; save Figure 9 and timings to outputs/7.6/.

from functools import wraps
from pathlib import Path
import argparse
import contextlib as _t76_contextlib
import csv as _t76_csv
import io as _t76_io
import json as _t76_json
import os as _t76_os
import pickle as _t76_pickle
import platform as _t76_platform
import statistics as _t76_statistics
import subprocess as _t76_subprocess
import sys as _t76_sys
import tempfile as _t76_tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import root


T76_shared = {}
T76_pending = None
T76_current = None
T76_stage = None

def t76_record():
    return dict.fromkeys(('T_total', 'T_setup_update', 'T_apply_solve', 'T_eig', 'T_cert',
        'T_shared_initial', 'T_H0_build', 'T_shared_spectrum', 'T_shared_other',
        'T_method_initial', 'T_initial_eig', 'T_outer', 'T_metric_setup',
        'T_metric_apply', 'T_metric_solve', 'T_frame', 'T_cert_Hessian',
        'initial_setup_count', 'setup_count', 'update_count', 'apply_count', 'solve_count',
        'state_updates', 'frame_iterations', 'factorization_attempt_count',
        'factorization_failure_count', 'certification_count'), 0)

def t76_begin_method():
    global T76_pending
    T76_pending = {'record': t76_record(), 'start': time.perf_counter()}

def t76_end_method(m):
    global T76_pending
    m['timing']['T_method_initial'] = time.perf_counter() - T76_pending['start']
    T76_pending = None

def t76_metric(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        t = T76_pending['record']
        metric = fn(*args, **kwargs)
        t['T_metric_setup'] = time.perf_counter() - T76_pending['start']
        t['T_setup_update'] = t['T_metric_setup']
        if metric['kind'] != 'I':
            t['initial_setup_count'] = t['setup_count'] = 1
            if metric['kind'] == 'sparse':
                solver = metric['solve'].__defaults__[0]
                owner = getattr(solver, '__self__', None)
                t['factorization_backend'] = 'scipy.sparse.linalg.factorized: ' + (type(owner).__name__ if owner is not None else type(solver).__name__)
            else:
                t['factorization_backend'] = 'scipy.linalg.cho_factor / cho_solve'
            for key, field, count in (('apply', 'T_metric_apply', 'apply_count'), ('solve', 'T_metric_solve', 'solve_count')):
                operation = metric[key]
                def measured(b, operation=operation, field=field, count=count):
                    tick = time.perf_counter()
                    try:
                        return operation(b)
                    finally:
                        elapsed = time.perf_counter() - tick
                        t[field] += elapsed
                        t[count] += 1
                metric[key] = measured
        metric['timing76'] = t
        return metric
    return wrapped

def t76_method(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        tick = time.perf_counter()
        m = fn(*args, **kwargs)
        elapsed = time.perf_counter() - tick
        m['timing'] = T76_pending['record']
        m['timing']['T_initial_eig'] = elapsed
        return m
    return wrapped

def t76_hessian(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        tick = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - tick
            if T76_stage == 'shared':
                T76_shared['T_H0_build'] = elapsed
            elif T76_stage == 'certification':
                T76_current['T_cert_Hessian'] += elapsed
    return wrapped

def t76_certificate(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global T76_stage
        T76_stage = 'certification'
        tick = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            T76_current['T_cert'] += time.perf_counter() - tick
            T76_current['certification_count'] += 1
            T76_stage = None
    return wrapped

# Total solver cost includes shared setup, method setup and iteration; certification is separate.
def t76_run(fn):
    @wraps(fn)
    def wrapped(m, *args, **kwargs):
        global T76_current, T76_outer_start
        T76_current = m['timing']
        T76_outer_start = time.perf_counter()
        output = fn(m, *args, **kwargs)
        t = m['timing']
        t.update(T76_shared)
        t['T_total'] = t['T_shared_initial'] + t['T_method_initial'] + t['T_outer']
        t['T_apply_solve'] = t['T_metric_apply'] + t['T_metric_solve']
        t['T_eig'] = t['T_H0_build'] + t['T_shared_spectrum'] + t['T_initial_eig'] + t['T_frame']
        output['timing'] = t.copy()
        T76_current = None
        return output
    return wrapped
# Discretization / model
H_GRID = 2.0**-8
N_GRID = 255
LAMBDA = 0.02

# Initial condition
U0_AMPLITUDE = 0.5
U0_MODE = 1

# Stopping / numerical differentiation
TOL = 1e-6
MAX_ITER = 2000000
PLOT_MAX_ITER = 300
HV_EPS = 1e-5

# State solve
STATE_TOL = 1e-10
STATE_MAXIT = 30

# Divergence safety
DIVERGE_GRAD = 1e8
DIVERGE_UINF = 1e4

# Standard HiSD steps
STANDARD_ETA = 1e-4
COMMON_TAU = 0.001
DIRECTION_ITERS = 5

# p-HiSD step size shared by all p-HiSD methods
PHISD_ETA = 1

# Method switches
INCLUDE_H1 = True
INCLUDE_SHIFTED_CHOLESKY = True
INCLUDE_FROZEN_SPECTRAL = True

# Shifted-Cholesky settings
SHIFT_FACTOR = 2.0

# Regularization / SPD shifts
EPS_HESSIAN_FACTOR = 1e-6
EPS_DENSE_METRIC_FACTOR = 1e-12

OUTPUT_FIG = "6.pdf"

h, N, lam = H_GRID, N_GRID, LAMBDA
x = np.linspace(h, 1.0 - h, N)
yd = -2.0 * np.sin(np.pi * x)
L = sp.diags([(-1 / h**2) * np.ones(N - 1), (2 / h**2) * np.ones(N), (-1 / h**2) * np.ones(N - 1)], [-1, 0, 1], format="csr")
I = sp.eye(N, format="csr")
LI = (L + I).tocsr()
H1 = LI.tocsr()
LABELS = {
    "standard_hisd": "HiSD",
    "p_hisd_H1": "Laplacian",
    "p_hisd_shifted_cholesky": "shifted Cholesky",
    "p_hisd_frozen_spectral": "Spectral",
}
IDM = {"kind": "I", "solve": lambda b: np.asarray(b, float), "apply": lambda b: np.asarray(b, float), "dense": np.eye(N)}

def rhs_control(u): return 0.001 * u**2 + np.cos(2 * np.pi * u)
def rhs_prime(u): return 0.002 * u - 2 * np.pi * np.sin(2 * np.pi * u)
def rhs_second(u): return 0.002 - 4 * (np.pi**2) * np.cos(2 * np.pi * u)

# Solve the nonlinear state equation by Newton iteration, with a root-solver fallback.
def solve_state(u, y0=None, tol=STATE_TOL, maxit=STATE_MAXIT):
    u, y = np.asarray(u, float), (np.zeros(N) if y0 is None else np.asarray(y0, float).copy())
    for _ in range(maxit):
        F = (L @ y) + y + y**3 - rhs_control(u)
        if np.linalg.norm(F, np.inf) < tol: return y
        J = (LI + sp.diags(3 * y**2, 0, format="csr")).tocsc(); dy = spla.spsolve(J, -F)
        if not np.all(np.isfinite(dy)): break
        y += dy
        if np.linalg.norm(dy, np.inf) <= tol * (1 + np.linalg.norm(y, np.inf)): return y
    sol = root(lambda yy: (L @ yy) + yy + yy**3 - rhs_control(u), y,
               jac=lambda yy: (LI + sp.diags(3 * yy**2, 0, format="csr")).toarray(), method="hybr", tol=tol)
    if not sol.success: raise RuntimeError(f"State solve failed: {sol.message}")
    return np.asarray(sol.x, float)

def solve_adjoint(y):
    A = (LI + sp.diags(3 * y**2, 0, format="csr")).tocsc()
    return np.asarray(spla.spsolve(A, y - yd), float)

# Eliminate the PDE state and use the adjoint to evaluate the reduced gradient.
def grad_full(u, y0=None):
    y = solve_state(u, y0=y0); p = solve_adjoint(y)
    return y, p, np.asarray(lam * (LI @ u) + p * rhs_prime(u), float)

def cost(u, y):
    e = y - yd
    return 0.5 * h * float(e @ e) + 0.5 * lam * h * float(u @ (LI @ u))

# Approximate the reduced Hessian action by centered gradient differences.
# Normalize the perturbation direction, then restore its magnitude in the result.
def hvp(u, v, eps, y_ref=None):
    v = np.asarray(v, float); nv = float(np.linalg.norm(v))
    if nv <= 1e-16: return np.zeros_like(v)
    vh = v / nv; y_ref = solve_state(u, y0=np.zeros(N)) if y_ref is None else y_ref
    gp = grad_full(u + eps * vh, y0=y_ref)[2]; gm = grad_full(u - eps * vh, y0=y_ref)[2]
    return nv * (gp - gm) / (2 * eps)

@t76_hessian
def dense_hessian(u, eps):
    H, E = np.zeros((N, N)), np.eye(N); yref = solve_state(u, y0=np.zeros(N))
    for j in range(N): H[:, j] = hvp(u, E[:, j], eps, y_ref=yref)
    return 0.5 * (H + H.T)

# Factor each fixed metric once; reuse its solve in every state and frame update.
@t76_metric
def make_metric(A=None, kind="I"):
    if kind == "I": return IDM
    if kind == "sparse":
        T76_pending['record']['factorization_attempt_count'] += 1
        Ac = A.tocsc(); solver = spla.factorized(Ac)
        return {"kind": "sparse", "solve": lambda b, s=solver: np.asarray(s(b), float), "apply": lambda b, M=Ac: np.asarray(M @ b, float), "dense": Ac.toarray()}
    A0 = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T); step, shift = EPS_DENSE_METRIC_FACTOR * max(1.0, float(np.linalg.norm(A0, 2))), 0.0
    while True:
        At = A0 + shift * np.eye(A0.shape[0])
        try:
            T76_pending['record']['factorization_attempt_count'] += 1
            cho = la.cho_factor(At, lower=True, check_finite=False)
            return {"kind": "dense", "solve": lambda b, c=cho: la.cho_solve(c, b, check_finite=False), "apply": lambda b, M=At: M @ b, "dense": At}
        except la.LinAlgError:
            T76_pending['record']['factorization_failure_count'] += 1
            shift = step if shift == 0.0 else 10.0 * shift
            if shift > 1e6: raise RuntimeError("Failed SPD dense metric")

def normalize(v, m):
    v = np.asarray(v, float)
    if m["kind"] == "I":
        n = float(np.linalg.norm(v))
        if n <= 1e-16: raise RuntimeError("Euclidean normalization failed")
        return v / n
    q = float(v @ m["apply"](v))
    if q <= 1e-18: raise RuntimeError("M-normalization failed")
    return v / np.sqrt(q)

def kappa(vals):
    a = np.abs(np.asarray(vals, float)); mx = float(np.max(a)); nz = a[a > 1e-12 * max(1.0, mx)]
    return (np.inf if nz.size == 0 else float(mx / np.min(nz))), mx

@t76_method
def make_method(name, metric, standard, H0, eta, tau, block_size=-1, alpha=np.nan):
    if standard:
        vals, vecs = np.linalg.eigh(H0); i = int(np.argmin(vals)); v0 = normalize(vecs[:, i], IDM); ge, Lm = np.asarray(vals, float), float(np.max(np.abs(vals)))
    else:
        ge, V = la.eigh(H0, metric["dense"]); i = int(np.argmin(ge)); v0 = normalize(V[:, i], metric); ge = np.asarray(ge, float); _, Lm = kappa(ge)
    kap, _ = kappa(ge)
    return {"name": name, "label": LABELS[name], "standard": standard, "solve": metric["solve"], "apply": metric["apply"], "v0": v0,
            "eta": float(eta), "tau": float(tau), "kappa_M0": float(kap), "L_M0": float(Lm), "block_size": int(block_size), "alpha": float(alpha)}

# Classify the ordinary reduced Hessian with a scale-dependent sign threshold.
# Near-zero modes are unresolved; this post-solver check does not drive the iteration.
@t76_certificate
def final_index(u, eps):
    ev = np.linalg.eigvalsh(dense_hessian(u, eps)); mx = float(np.max(np.abs(ev))); th = 1e-6 * max(1.0, mx)
    return int(np.sum(ev < -th)), int(np.sum(np.abs(ev) <= th)), float(np.min(ev)), mx

# Stop on the raw reduced-gradient norm or a safeguard; record the final index separately.
@t76_run
def run(m, u0, tol, max_iter, hv_eps, direction_iters):
    u, v = np.asarray(u0, float).copy(), np.asarray(m["v0"], float).copy()
    eta, tau, t0, hist, status, conv = m["eta"], m["tau"], time.perf_counter(), [], "max_iter", False
    y = np.zeros_like(u)
    for it in range(max_iter + 1):
        y, _, g = grad_full(u); gn, J = float(np.linalg.norm(g)), float(cost(u, y))
        hist.append({"iter": it, "time": float(time.perf_counter() - t0), "grad_norm": gn})
        if (not np.isfinite(gn)) or (not np.isfinite(J)) or gn > DIVERGE_GRAD or float(np.linalg.norm(u, np.inf)) > DIVERGE_UINF: status = "diverged"; break
        if gn <= tol: status, conv = "converged", True; break
        if it == max_iter: break
        # Perform direction_iters frame sweeps at u before the reflected control update.
        if m["standard"]:
            t76_frame_start = time.perf_counter()
            for _ in range(direction_iters):
                Hv = hvp(u, v, hv_eps, y_ref=y)
                ray = float(v @ Hv)
                v = normalize(v - tau * (Hv - v * ray), IDM)
            T76_current['T_frame'] += time.perf_counter() - t76_frame_start
            T76_current['frame_iterations'] += direction_iters
            d = -g + 2 * v * float(v @ g)
        else:
            t76_frame_start = time.perf_counter()
            for _ in range(direction_iters):
                Hv = hvp(u, v, hv_eps, y_ref=y)
                z = m["solve"](Hv)
                ray = float(v @ Hv)
                v = normalize(v - tau * (z - v * ray), {"kind": "M", "apply": m["apply"]})
            T76_current['T_frame'] += time.perf_counter() - t76_frame_start
            T76_current['frame_iterations'] += direction_iters
            d = -m["solve"](g) + 2 * v * float(v @ g)
        u = u + eta * d
        T76_current['state_updates'] += 1
        if (not np.all(np.isfinite(u))) or (not np.all(np.isfinite(v))): status = "diverged"; break
    T76_current['T_outer'] = time.perf_counter() - T76_outer_start
    fi, _, _, _ = final_index(u, hv_eps)
    return {"method": m["name"], "hist": hist, "u": u, "y": y, "grad": gn, "J": J, "iter": it, "status": status, "converged": conv, "final_index": fi, "time": float(hist[-1]["time"])}

def fmt_kappa(k): return f"{k:.2e}" if k >= 100 else f"{k:.2f}"

def plot_grad(outputs, tol, path, plot_max_iter):
    plt.figure(figsize=(8, 5.5))
    for o in outputs:
        it = np.array([r["iter"] for r in o["hist"]], float); gn = np.array([r["grad_norm"] for r in o["hist"]], float)
        mk = it < float(plot_max_iter)
        it, gn = it[mk], gn[mk]
        if it.size == 0: continue
        plt.semilogy(it, gn, lw=2.0, label=f"{LABELS[o['method']]}")
    plt.xlabel("Iteration $m$", fontsize=17); plt.ylabel(r"$\|\nabla \hat{J}(u_m)\|_2$", fontsize=17); plt.yscale("log"); plt.title("Non-convex Optimal Control Problem", fontsize=17)
    plt.tick_params(axis="both", labelsize=17)
    plt.grid(True, ls="--", alpha=0.5); plt.legend(fontsize=15, loc="upper right", bbox_to_anchor=(1, 0.6)); plt.tight_layout(); plt.savefig(path, format="pdf", bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser(description="OCP Example 4.1")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--plot-max-iter", type=int, default=PLOT_MAX_ITER)
    ap.add_argument("--hv-eps", type=float, default=HV_EPS)
    ap.add_argument(
        "--output-dir",
        type=str,
        default=str((
            Path(__file__).resolve().parent
            / ".." / ".." / "outputs" / "7.6"
        ).resolve())
    )
    a = ap.parse_args()
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not _t76_is_child:
        _t76_formal(a, outdir)
        return

    print("Active configuration:")
    print(f"  N = {N_GRID}")
    print(f"  h = {H_GRID}")
    print(f"  lambda = {LAMBDA}")
    print(f"  u0 amplitude = {U0_AMPLITUDE}")
    print(f"  tol = {a.tol}")
    print(f"  max_iter = {a.max_iter}")
    print(f"  plot_max_iter = {a.plot_max_iter}")
    print(f"  hv_eps = {a.hv_eps}")
    print(f"  standard eta = {STANDARD_ETA}")
    print(f"  common tau = {COMMON_TAU}")
    print(f"  direction_iters = {DIRECTION_ITERS}")
    print(f"  p-HiSD eta = {PHISD_ETA}")
    print(f"  include H1/shifted-Chol/spectral = {INCLUDE_H1}/{INCLUDE_SHIFTED_CHOLESKY}/{INCLUDE_FROZEN_SPECTRAL}")
    print(f"  shifted factor = {SHIFT_FACTOR}")

    u0 = U0_AMPLITUDE * np.sin(2 * np.pi * U0_MODE * x); _, _, g0 = grad_full(u0)
    print("Building dense reduced Hessian H0 ...")
    global T76_shared, T76_stage
    T76_shared = {}
    T76_stage = 'shared'
    t76_shared_start = time.perf_counter()
    # The initial reduced Hessian supplies all initial frames and the two Hessian-based metrics.
    H0 = dense_hessian(u0, a.hv_eps)
    t76_shared_spectrum_start = time.perf_counter()
    eig_ord, Q0 = np.linalg.eigh(H0)
    T76_shared['T_shared_spectrum'] = time.perf_counter() - t76_shared_spectrum_start
    print(f"Initial ||grad(u0)||_2 = {float(np.linalg.norm(g0)):.6e}")
    print(f"Initial ordinary negative count = {int(np.sum(eig_ord < 0.0))}")

    lam_min = float(np.min(eig_ord))
    h0_norm = float(np.max(np.abs(eig_ord)))
    T76_shared['T_shared_initial'] = time.perf_counter() - t76_shared_start
    T76_shared['T_shared_other'] = T76_shared['T_shared_initial'] - T76_shared['T_H0_build'] - T76_shared['T_shared_spectrum']
    T76_stage = None
    t76_begin_method()
    methods = [make_method("standard_hisd", IDM, True, H0, eta=STANDARD_ETA, tau=COMMON_TAU)]
    t76_end_method(methods[-1])
    if INCLUDE_H1:
        t76_begin_method()
        methods.append(make_method("p_hisd_H1", make_metric(H1, "sparse"), False, H0, eta=PHISD_ETA, tau=COMMON_TAU))
        t76_end_method(methods[-1])
    if INCLUDE_SHIFTED_CHOLESKY:
        t76_begin_method()
        delta = SHIFT_FACTOR * max(0.0, -lam_min) + EPS_HESSIAN_FACTOR * max(1.0, h0_norm)
        M_shifted_chol = 0.5 * (H0 + delta * np.eye(N) + (H0 + delta * np.eye(N)).T)
        methods.append(
            make_method(
                "p_hisd_shifted_cholesky",
                make_metric(M_shifted_chol, "dense"),
                False,
                H0,
                eta=PHISD_ETA,
                tau=COMMON_TAU,
            )
        )
        t76_end_method(methods[-1])
    if INCLUDE_FROZEN_SPECTRAL:
        t76_begin_method()
        eps_spec = 1e-2
        M_spec = Q0 @ np.diag(np.abs(eig_ord) + eps_spec) @ Q0.T
        M_spec = 0.5 * (M_spec + M_spec.T)
        methods.append(
            make_method(
                "p_hisd_frozen_spectral",
                make_metric(M_spec, "dense"),
                False,
                H0,
                eta=PHISD_ETA,
                tau=COMMON_TAU,
            )
        )
        t76_end_method(methods[-1])
    print("\nMethod step setup:")
    for m in methods:
        print(f"  {LABELS[m['name']]}: eta={m['eta']:.6g}, tau={m['tau']:.6g}, direction_iters={DIRECTION_ITERS}, kappa_M0={fmt_kappa(m['kappa_M0'])}, L_M0={m['L_M0']:.2e}")
    print(f"\nRuntime policy: max_iter budget={a.max_iter}, one state update per outer iteration, {DIRECTION_ITERS} direction updates per outer iteration")

    outs = [run(m, u0, a.tol, a.max_iter, a.hv_eps, DIRECTION_ITERS) for m in methods]

    print("\nMethod | kappa_M0 | iterations | final_grad | final_index | status")
    mm = {m["name"]: m for m in methods}
    for o in outs:
        print(f"{LABELS[o['method']]}: kappa_M0={fmt_kappa(mm[o['method']]['kappa_M0'])}, iter={o['iter']}, grad={o['grad']:.2e}, index={o['final_index']}, status={o['status']}")

    if _t76_skip_plot:
        global _t76_captured
        _t76_captured = locals().copy()
        return
    fig = outdir / OUTPUT_FIG; plot_grad(outs, a.tol, fig, a.plot_max_iter)
    print("\nSaved file:")
    print(f"  {fig}")
    print(f"  Plot includes iter < {a.plot_max_iter}")
    global TIMING76_RESULT
    TIMING76_RESULT = locals().copy()


_t76_is_child = False
_t76_skip_plot = False
_t76_overlap = 'component timing categories must not be summed blindly to obtain T_total.'
_t76_definitions = {
    'T_total': 'Assembled standalone method estimate = T_shared_initial + T_method_initial + T_outer; not a contiguous stopwatch. Shared initialization is executed once in its original position and charged once to each method. Python startup, common problem/u0/g0 construction, certification, plotting, summaries and file I/O are excluded.',
    'T_shared_initial': 'Original interval from before H0=dense_hessian through h0_norm: initial dense Hessian, shared ordinary eigendecomposition, original diagnostics, lam_min and h0_norm.',
    'T_H0_build': 'Complete original shared dense_hessian call, including finite-difference HVP and state/adjoint work.',
    'T_shared_spectrum': 'Original shared np.linalg.eigh(H0). The subsequent make_method eigendecomposition is unchanged.',
    'T_shared_other': 'T_shared_initial minus T_H0_build and T_shared_spectrum, including original diagnostics and instrumentation overhead.',
    'T_method_initial': 'Original method preparation from t76_begin_method to t76_end_method, including metric construction/setup and initial eigenspace.',
    'T_setup_update': 'Equal to T_metric_setup: actual initial metric construction/factorization only. H1 sparse conversion/factorized; shifted-Cholesky delta/construction/factorization; frozen-spectral eps_spec/construction/factorization. Common PDE matrices excluded. No metric updates; update_count=0.',
    'T_metric_setup': 'Existing initial metric setup interval measured by t76_metric from the method preparation start. Standard identity has zero.',
    'T_apply_solve': 'T_metric_apply + T_metric_solve, including only actual explicit metric callable executions. No added metric operations; Standard identity has zero.',
    'T_metric_apply': 'Original explicit metric apply callable executions, including initial M-normalization and frame normalizations.',
    'T_metric_solve': 'Original explicit metric solve callable executions in state directions and frame updates.',
    'T_eig': 'Inclusive diagnostic = T_H0_build + T_shared_spectrum + T_initial_eig + T_frame. Includes nested metric work also recorded in T_apply_solve.',
    'T_initial_eig': 'Original make_method including ordinary/generalized eigensolver, selection, normalization and kappa/L_M0. Internal generalized-eigensolver work remains here.',
    'T_frame': 'Original direction-update blocks, including HVP/state/adjoint work, Rayleigh/projection, normalization and nested metric apply/solve.',
    'T_outer': 'Original run entry through the end of the unchanged outer loop, stopping before final_index. Includes residual/cost/history/state/frame work and timing overhead.',
    'T_cert': 'Only the original final_index call: endpoint dense Hessian, eigvalsh and Morse-index threshold/count computation. Outside T_total.',
    'T_cert_Hessian': 'Original dense_hessian subinterval within final_index, included in T_cert only.',
    'legacy_history_time': 'Original output time and per-point history clocks retained separately; excluded from numerical equality checks.',
    'nonconverged_T_total': 'Consumed wall-clock until termination, not time-to-solution. No speedup is reported for nonconverged methods.',
}


def _t76_reference_source(source):
    import ast
    tree = ast.parse(source.read_text(encoding='utf-8'))

    class ReferenceSolver(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name.startswith('_t76_') or node.name == '_publication_data':
                return None
            return self.generic_visit(node)

        def visit_Import(self, node):
            node.names = [item for item in node.names
                          if not (item.asname or '').startswith('_t76_')]
            return node if node.names else None

        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id.startswith('_t76_')
                   for target in node.targets):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            names = {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
            if names & {'_t76_is_child', '_t76_skip_plot', '_t76_sys'}:
                return None
            return self.generic_visit(node)

    tree = ReferenceSolver().visit(tree)
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree) + '\n').encode('utf-8')


def _t76_plain(value):
    if isinstance(value, dict):
        return {key: _t76_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_t76_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _t76_plain(value.tolist())
    if isinstance(value, np.generic):
        return _t76_plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _t76_environment():
    import scipy
    import matplotlib
    stream = _t76_io.StringIO()
    with _t76_contextlib.redirect_stdout(stream):
        np.show_config()
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
    return dict(python=_t76_sys.version, executable=_t76_sys.executable,
                numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                platform=_t76_platform.platform(), machine=_t76_platform.machine(),
                thread_environment={key: _t76_os.environ.get(key) for key in keys},
                numpy_configuration=stream.getvalue())


def _t76_verify(row, direction_iters):
    t = row['timing']
    for key, value in t.items():
        if key.startswith('T_') and (not np.isfinite(value) or value < 0):
            raise RuntimeError('Invalid timer: ' + key)
    equations = (
        (t['T_total'], t['T_shared_initial'] + t['T_method_initial'] + t['T_outer']),
        (t['T_shared_initial'], t['T_H0_build'] + t['T_shared_spectrum'] + t['T_shared_other']),
        (t['T_eig'], t['T_H0_build'] + t['T_shared_spectrum'] + t['T_initial_eig'] + t['T_frame']),
        (t['T_apply_solve'], t['T_metric_apply'] + t['T_metric_solve']),
        (t['T_setup_update'], t['T_metric_setup']),
    )
    if not all(np.isclose(a, b, rtol=1e-12, atol=1e-10) for a, b in equations):
        raise RuntimeError('Timing definition mismatch')
    if t['update_count'] != 0 or t['certification_count'] != 1:
        raise RuntimeError('Setup/certification count mismatch')
    if t['frame_iterations'] != direction_iters * t['state_updates']:
        raise RuntimeError('Frame count mismatch')
    if row['status'] != 'diverged' and (t['state_updates'] != row['iterations'] or row['residual_points'] != row['iterations'] + 1):
        raise RuntimeError('Iteration count mismatch')
    if row['method'] == 'standard_hisd':
        if any(t[key] != 0 for key in ('T_setup_update', 'T_apply_solve', 'setup_count', 'initial_setup_count', 'apply_count', 'solve_count')):
            raise RuntimeError('Unexpected Standard preconditioner operation')
    elif (t['setup_count'] != 1 or t['initial_setup_count'] != 1
          or t['apply_count'] != 1 + direction_iters * t['state_updates']
          or t['solve_count'] != (direction_iters + 1) * t['state_updates']):
        raise RuntimeError('Metric operation count mismatch')
    if t['T_cert_Hessian'] > t['T_cert'] + 1e-8:
        raise RuntimeError('Certification subinterval mismatch')


def _t76_capture(ns, local):
    parameters = {key: value for key, value in ns.items()
                  if key.isupper() and not key.startswith(('T76', 'TIMING76'))
                  and isinstance(value, (int, float, bool, str))}
    parameters['effective_arguments'] = {key: value for key, value in vars(local['a']).items() if key != 'output_dir'}
    initial_keys = ('u0', 'g0', 'H0', 'eig_ord', 'Q0', 'lam_min', 'h0_norm',
                    'delta', 'M_shifted_chol', 'eps_spec', 'M_spec')
    numerical = dict(parameters=parameters, initial={key: local[key] for key in initial_keys if key in local},
                     methods=[], outputs=[])
    rows, clocks = [], []
    for method, output in zip(local['methods'], local['outs']):
        m = {key: value for key, value in method.items() if key not in ('apply', 'solve', 'timing')}
        if method['standard']:
            metric = ns['IDM']['dense']
        else:
            operation = method['apply'].__defaults__[0]
            metric = operation.__defaults__[0]
            if sp.issparse(metric):
                metric = metric.toarray()
        m['metric_dense'] = metric.copy()
        numerical['methods'].append(m)
        o = {key: value for key, value in output.items() if key not in ('time', 'timing', 'hist')}
        o['history_iter'] = np.array([point['iter'] for point in output['hist']], dtype=np.int64)
        o['history_residual'] = np.array([point['grad_norm'] for point in output['hist']], dtype=np.float64)
        numerical['outputs'].append(o)
        clocks.append(np.array([point['time'] for point in output['hist']], dtype=np.float64))
        row = dict(method=method['name'], label=method['label'], status=output['status'],
                   converged=output['converged'], iterations=output['iter'], residual_points=len(output['hist']),
                   final_residual=output['grad'], final_cost_J=output['J'], final_index=output['final_index'],
                   kappa_M0=method['kappa_M0'], L_M0=method['L_M0'],
                   legacy_outer_history_time=output['time'], timing=output['timing'].copy())
        _t76_verify(row, ns['DIRECTION_ITERS'])
        rows.append(row)
    return dict(numerical=numerical, methods=rows, legacy_history_times=clocks,
                shared_timing=ns['T76_shared'].copy(), environment=_t76_environment(),
                process_id=_t76_os.getpid(), output_dir=str(local['outdir'].resolve()))


def _t76_compare(reference, candidate):
    failures, checked = [], 0
    maximum = 0.0
    def check(a, b, path):
        nonlocal checked, maximum
        checked += 1
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                failures.append(path + ': keys')
            else:
                for key in a:
                    check(a[key], b[key], path + '.' + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) != type(b) or len(a) != len(b):
                failures.append(path + ': sequence')
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    check(x, y, path + '[' + str(i) + ']')
        elif isinstance(a, np.ndarray):
            if not (isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)):
                failures.append(path + ': array')
                if isinstance(b, np.ndarray) and a.shape == b.shape and a.size:
                    delta = np.abs(a - b)
                    maximum = max(maximum, float(np.max(delta[np.isfinite(delta)])) if np.any(np.isfinite(delta)) else float('inf'))
        elif not (a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))):
            failures.append(path + ': scalar')
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                maximum = max(maximum, abs(float(a) - float(b)))
    check(reference['numerical'], candidate['numerical'], 'numerical')
    check(reference['environment'], candidate['environment'], 'environment')
    counts = lambda result: [{key: value for key, value in row['timing'].items() if not key.startswith('T_')} for row in result['methods']]
    check(counts(reference), counts(candidate), 'operation_counts_and_backend')
    if not reference['rng_state_unchanged'] or not candidate['rng_state_unchanged']:
        failures.append('Unexpected RNG consumption')
    return dict(passed=not failures, comparison='Exact equality; numpy.array_equal(equal_nan=True), including shape and dtype',
                maximum_numerical_difference=maximum, checked_objects=checked, failures=failures,
                scope='All parameters, shared initialization, initial frames and dense metrics, complete iteration/residual histories, returned u/y/gradient norm/J/index/status, operation counts and backend; clocks excluded.')


def _t76_child():
    global _t76_is_child, _t76_skip_plot
    mode, artifact, baseline = _t76_sys.argv[2:5]
    _t76_sys.argv = [_t76_sys.argv[0]] + _t76_sys.argv[5:]
    _t76_is_child = True
    _t76_skip_plot = mode != 'formal1'
    rng_before = np.random.get_state()
    if mode == 'baseline':
        ns = {'__file__': baseline, '__name__': '_section76_reference'}
        exec(compile(Path(baseline).read_bytes(), baseline, 'exec'), ns)
        ns['main']()
        local = ns['TIMING76_RESULT']
    else:
        main()
        ns = globals()
        local = ns.get('_t76_captured', ns.get('TIMING76_RESULT'))
    rng_after = np.random.get_state()
    result = _t76_capture(ns, local)
    result['rng_state_unchanged'] = all(np.array_equal(a, b) for a, b in zip(rng_before, rng_after))
    result['mode'] = mode
    with Path(artifact).open('wb') as stream:
        _t76_pickle.dump(result, stream, protocol=5)


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


def _t76_write_json(path, value):
    path.write_text(_t76_json.dumps(_publication_data(_t76_plain(value)), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _t76_write_outputs(outdir, records, regression, checks):
    summaries, arrays = [], {}
    for i, first in enumerate(records[0]['methods']):
        timing = {}
        for key in first['timing']:
            if key.startswith('T_'):
                values = [record['methods'][i]['timing'][key] for record in records]
                timing[key] = dict(zip(('repetition_1', 'repetition_2', 'repetition_3'), values))
                timing[key].update(median=_t76_statistics.median(values), min=min(values), max=max(values))
        summaries.append(dict(method=first['method'], label=first['label'], status=first['status'],
                              converged=first['converged'], iterations=first['iterations'],
                              final_residual=first['final_residual'], final_index=first['final_index'],
                              timing=timing,
                              operation_counts_and_backend={key: value for key, value in first['timing'].items() if not key.startswith('T_')}))
    def pack(value, path):
        if isinstance(value, np.ndarray):
            arrays[path] = value
            return dict(npz_key=path, shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, dict):
            return {key: pack(item, path + '.' + key) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item, path + '.' + str(i)) for i, item in enumerate(value)]
        return value
    raw = [pack(record, 'repetition_' + str(i)) for i, record in enumerate(records, 1)]
    paths = {name: str((outdir / name).resolve()) for name in
             (OUTPUT_FIG, 'timing76_raw.json', 'timing76_raw.npz', 'timing76_summary.json', 'timing76_summary.csv', 'timing76_audit.md')}
    metadata = dict(timing_repeats=3, time_unit='seconds', retained_experiment_repetition=1,
                    T_total_includes_certification=False, T_total_is_assembled_standalone_estimate=True,
                    component_overlap=_t76_overlap, timing_definitions=_t76_definitions,
                    fresh_processes=True,
                    solver_warmup=False, parameter_tuning=False, environment_changes=False,
                    method_order=[row['method'] for row in records[0]['methods']],
                    numerical_regression=regression,
                    formal_checks_against_instrumented_reference=checks,
                    repetition_2_3_pdf_unchanged=True,
                    output_dir=str(outdir.resolve()), paths=paths)
    np.savez_compressed(outdir / 'timing76_raw.npz', **arrays)
    _t76_write_json(outdir / 'timing76_raw.json', dict(metadata=metadata, repetitions=raw))
    _t76_write_json(outdir / 'timing76_summary.json', dict(metadata=metadata, methods=summaries))
    columns = ['method', 'label', 'status', 'converged', 'iterations', 'final_residual', 'final_index', 'quantity',
               'repetition_1', 'repetition_2', 'repetition_3', 'median', 'min', 'max']
    with (outdir / 'timing76_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = _t76_csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for method in summaries:
            for quantity, stats in method['timing'].items():
                row = {key: method[key] for key in columns[:7]}
                writer.writerow(dict(row, quantity=quantity, **stats))
    lines = ['# Section 7.6 results and timing', '',
             'Three independent repetitions are summarized below. Times are in seconds.', '',
             '## Numerical results', '',
             '| Method | Iterations | Status | Final residual | Index | Endpoint max diff | Full-history max diff |',
             '|---|---:|---|---:|---:|---:|---:|',]
    for method in summaries:
        lines.append(f"| {method['label']} | {method['iterations']} | {method['status']} | {method['final_residual']:.17g} | {method['final_index']} | 0 | 0 |")
    lines += ['', 'Operation counts and factorization backend (identical across baseline, reference and all formal repetitions):', '']
    for method in summaries:
        lines.append('- ' + method['label'] + ': `' + _t76_json.dumps(method['operation_counts_and_backend'], sort_keys=True) + '`')
    lines += ['', '## Timing', '', 'timing_repeats = 3', '',
              'All values are seconds. Use median in the manuscript. Each row retains all three raw times in JSON/CSV. No solver warm-up, tuning, backend or thread changes are applied.', '',
              '| Method | Quantity | Median | Min | Max |', '|---|---|---:|---:|---:|']
    for method in summaries:
        for quantity, stats in method['timing'].items():
            lines.append(f"| {method['label']} | {quantity} | {stats['median']:.9g} | {stats['min']:.9g} | {stats['max']:.9g} |")
    lines += ['', 'T_total is an assembled standalone method estimate. T_cert outside T_total.', _t76_overlap,
              'For nonconverged methods T_total is consumed wall-clock until termination, not time-to-solution; no speedup is reported.', '']
    for key, definition in _t76_definitions.items():
        lines.append('- `' + key + '`: ' + definition)
    lines += ['', '## Output files', '',
              'Files below are relative to this report. The default location is outputs/7.6/.', '']
    lines.extend('- `' + name + '`' for name in paths)
    lines += ['', '6.pdf uses the first formal repetition.',
              'Endpoint and full-history differences above compare the formal runs with the reference run.', '']
    (outdir / 'timing76_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    for method in summaries:
        stat = method['timing']['T_total']
        print(f"{method['label']}: {method['status']}, T_total median={stat['median']:.6f}s [{stat['min']:.6f}, {stat['max']:.6f}]", flush=True)
    print('Saved timing raw values, summaries and audit to ' + str(outdir.resolve()), flush=True)


# Check numerical agreement across fresh processes, then summarize three timing repetitions.
def _t76_formal(a, outdir):
    source = Path(__file__).resolve()
    for name in ('timing76_raw.json', 'timing76_raw.npz', 'timing76_summary.json', 'timing76_summary.csv'):
        (outdir / name).unlink(missing_ok=True)
    try:
        baseline_bytes = _t76_reference_source(source)
        source_bytes = source.read_bytes()
        with _t76_tempfile.TemporaryDirectory(prefix='.timing76-', dir=outdir.resolve()) as folder:
            work = Path(folder)
            baseline_path = work / 'reference_solver.py'
            baseline_path.write_bytes(baseline_bytes)
            baseline_path.chmod(0o444)
            effective = ['--tol', str(a.tol), '--max-iter', str(a.max_iter),
                         '--plot-max-iter', str(a.plot_max_iter), '--hv-eps', str(a.hv_eps)]
            def execute(mode, destination):
                if source.read_bytes() != source_bytes:
                    raise RuntimeError('run.py changed while the workflow was running')
                artifact = work / (mode + '.pickle')
                command = [_t76_sys.executable, '-B', str(source), '--timing-child', mode,
                           str(artifact), str(baseline_path), *effective, '--output-dir', str(destination.resolve())]
                print('Section 7.6 fresh process: ' + mode, flush=True)
                completed = _t76_subprocess.run(command)
                if completed.returncode:
                    raise RuntimeError(mode + ' child failed with exit code ' + str(completed.returncode))
                with artifact.open('rb') as stream:
                    return _t76_pickle.load(stream)
            baseline = execute('baseline', work / 'baseline')
            reference = execute('reference', work / 'reference')
            regression = _t76_compare(baseline, reference)
            if not regression['passed']:
                raise RuntimeError('Baseline numerical regression failed: ' + _t76_json.dumps(regression))
            print('Baseline regression PASS: exact equality, maximum difference 0', flush=True)
            records, checks, pdf_bytes = [], [], None
            timing_repeats = 3
            for repetition in range(1, timing_repeats + 1):
                record = execute('formal' + str(repetition), outdir)
                check = _t76_compare(reference, record)
                if not check['passed']:
                    raise RuntimeError('Formal repetition mismatch: ' + _t76_json.dumps(check))
                record['repetition'] = repetition
                current_pdf_bytes = (outdir / OUTPUT_FIG).read_bytes()
                if repetition == 1:
                    pdf_bytes = current_pdf_bytes
                elif current_pdf_bytes != pdf_bytes:
                    raise RuntimeError('Retained repetition-1 PDF was overwritten')
                records.append(record)
                checks.append(dict(repetition=repetition, **check))
                print(f'Formal repetition {repetition}/3: exact regression PASS', flush=True)
            if len({r['process_id'] for r in [baseline, reference] + records}) != 5:
                raise RuntimeError('Expected five distinct baseline/reference/formal process IDs')
            _t76_write_outputs(outdir, records, regression, checks)
    except Exception as error:
        for name in ('timing76_raw.json', 'timing76_raw.npz', 'timing76_summary.json', 'timing76_summary.csv'):
            (outdir / name).unlink(missing_ok=True)
        (outdir / 'timing76_audit.md').write_text(
            '# Section 7.6 timing audit — FAILED\n\nNo formal timing summary is valid for this invocation.\n\n'
            + _publication_data(str(error)) + '\n', encoding='utf-8')
        raise
if __name__ == "__main__":
    if len(_t76_sys.argv) > 1 and _t76_sys.argv[1] == '--timing-child':
        _t76_child()
        raise SystemExit(0)
    main()
