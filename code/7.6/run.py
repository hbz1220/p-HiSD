#!/usr/bin/env python3
"""
Merged single-file script:
- Experiment A: fixed-N convergence figure (from H1.py behavior)
- Experiment B: mesh-independence table (from H1_mesh.py behavior)
Run:
python /Users/huangbingzhang/Documents/code/python/final.py
"""

import csv
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu

# ============================================================
# Shared algorithmic parameters
# ============================================================
SHARED_TAU_V = 1e-4
SHARED_J_INNER = 5
PHISD_ETA_X = 0.5

# ============================================================
# Experiment A: fixed-N convergence figure parameters
# ============================================================
A_N = 128
A_P_LIST = [3, 5]
A_AMP = 1.2
A_ALPHA = 1.0
A_TOL = 1e-6

# Standard HiSD fixed parameters; manually tune A_STD_ETA_X if needed.
A_STD_ETA_X = 1e-4
A_STD_TAU_V = SHARED_TAU_V
A_STD_J_INNER = SHARED_J_INNER
A_STD_MAX_ITER = 60000

# p-HiSD-H1 fixed parameters
A_H1_ETA_X = PHISD_ETA_X
A_H1_TAU_V = SHARED_TAU_V
A_H1_J_INNER = SHARED_J_INNER
A_H1_MAX_ITER = 3000
A_DIVERGE_GRAD_NORM = 1e8
A_DIVERGE_ENERGY_ABS = 1e12
A_EIGSH_TOL = 1e-8
A_EIGSH_MAXITER = 3000
A_INDEX_EIGS_K = 6
A_INDEX_NEG_TOL = -1e-8
A_PLOT_FIRST_M = 100
A_OUT_SUMMARY_CSV = "H1_cp_summary.csv"
A_OUT_FIRST300_PDF = "6.pdf"
A_OUT_HISTORIES_NPZ = "H1_cp_histories.npz"
A_OUT_FULL_DEBUG_PDF = "H1_cp_full_convergence_debug.pdf"
A_OUT_H1_ZOOM_PDF = "H1_cp_H1_zoom.pdf"
A_OUT_SADDLE_PDF = "H1_cp_final_saddles.pdf"

# ============================================================
# Experiment B: mesh-independence table parameters
# ============================================================
B_P_LIST = [3, 5]
B_N_LIST = [64, 128, 192, 256]
B_AMP = 1.2
B_ALPHA = 1.0
B_TOL = 1e-6

B_H1_ETA_X = PHISD_ETA_X
B_H1_TAU_V = SHARED_TAU_V
B_H1_J_INNER = SHARED_J_INNER
B_H1_MAX_ITER = 5000
B_DIVERGE_GRAD_NORM = 1e8
B_DIVERGE_ENERGY_ABS = 1e12
B_EIGSH_TOL = 1e-8
B_EIGSH_MAXITER = 3000
B_INDEX_EIGS_K = 6
B_INDEX_NEG_TOL = -1e-8
B_OUT_SUMMARY_CSV = "H1_mesh_summary.csv"
B_OUT_HISTORIES_NPZ = "H1_mesh_histories.npz"

RANDOM_SEED = 20260502

A_FIELDS = [
    "p", "N", "n", "h", "amp", "method", "eta_x", "tau_v", "J", "max_iter_budget", "status", "iter",
    "initial_residual", "final_residual", "relative_final_residual", "lambda1", "lambda2", "index", "init_lambda",
    "setup_time", "iter_time", "time_total", "factor_time", "init_eig_time", "index_verify_time", "u_min", "u_max", "u_l2", "note",
]
B_FIELDS = [
    "p", "N", "n", "h", "amp", "alpha", "method", "eta_x", "tau_v", "J", "max_iter_budget", "status", "iter",
    "initial_grad_norm", "final_grad_norm", "relative_final_grad_norm", "lambda1", "lambda2", "index", "init_lambda",
    "setup_time", "iter_time", "time_total", "factor_time", "init_eig_time", "index_verify_time", "u_min", "u_max", "u_l2", "note",
]

A_STD_CFG = {
    "tol": A_TOL,
    "J": A_STD_J_INNER,
    "eigsh_tol": A_EIGSH_TOL,
    "eigsh_maxiter": A_EIGSH_MAXITER,
    "index_k": A_INDEX_EIGS_K,
    "index_neg_tol": A_INDEX_NEG_TOL,
    "div_grad": A_DIVERGE_GRAD_NORM,
    "div_E": A_DIVERGE_ENERGY_ABS,
}
A_CFG = {
    "tol": A_TOL,
    "J": A_H1_J_INNER,
    "eigsh_tol": A_EIGSH_TOL,
    "eigsh_maxiter": A_EIGSH_MAXITER,
    "index_k": A_INDEX_EIGS_K,
    "index_neg_tol": A_INDEX_NEG_TOL,
    "div_grad": A_DIVERGE_GRAD_NORM,
    "div_E": A_DIVERGE_ENERGY_ABS,
    "h1_tau": A_H1_TAU_V,
    "h1_eta": A_H1_ETA_X,
    "h1_max_iter": A_H1_MAX_ITER,
}
B_CFG = {
    "tol": B_TOL,
    "J": B_H1_J_INNER,
    "eigsh_tol": B_EIGSH_TOL,
    "eigsh_maxiter": B_EIGSH_MAXITER,
    "index_k": B_INDEX_EIGS_K,
    "index_neg_tol": B_INDEX_NEG_TOL,
    "div_grad": B_DIVERGE_GRAD_NORM,
    "div_E": B_DIVERGE_ENERGY_ABS,
    "h1_tau": B_H1_TAU_V,
    "h1_eta": B_H1_ETA_X,
    "h1_max_iter": B_H1_MAX_ITER,
}


def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def build_laplacian_2d(N):
    h = np.pi / (N + 1)
    e = np.ones(N)
    T = sp.diags([-e, 4.0 * e, -e], [-1, 0, 1], shape=(N, N), format="csr")
    I = sp.eye(N, format="csr")
    S = sp.diags([-e, -e], [-1, 1], shape=(N, N), format="csr")
    return ((sp.kron(I, T, format="csr") + sp.kron(S, I, format="csr")) / (h**2)).tocsr()


def make_u0(N, amp):
    h = np.pi / (N + 1)
    x = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    return (amp * np.sin(X) * np.sin(Y)).reshape(-1), h


def energy(u, A, p):
    return 0.5 * float(u @ (A @ u)) - (1.0 / (p + 1.0)) * float(np.sum(u ** (p + 1)))


def grad(u, A, p):
    return A @ u - u**p


def hess(u, A, p):
    return A - sp.diags(p * (u ** (p - 1)), offsets=0, format="csr")


def residual_h(g, h):
    return float(h * np.linalg.norm(g))


def normalize_l2(v):
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        raise ValueError("normalize_l2 got non-positive norm")
    return v / nrm


def normalize_M(v, M_dot):
    denom = float(v @ M_dot(v))
    if denom <= 0:
        raise ValueError(f"normalize_M got non-positive M-norm: {denom}")
    return v / np.sqrt(denom)


def merge_notes(*notes):
    return " | ".join(s.strip() for s in notes if isinstance(s, str) and s.strip())


def save_csv(rows, fields, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def verify_index(u, A, p, cfg):
    t0 = time.perf_counter()
    if not np.all(np.isfinite(u)):
        return np.nan, np.nan, -1, "index verification failed: final u has non-finite entries", time.perf_counter() - t0
    try:
        vals = eigsh(hess(u, A, p), k=cfg["index_k"], which="SA", tol=cfg["eigsh_tol"], maxiter=cfg["eigsh_maxiter"], return_eigenvectors=False)
        vals = np.sort(vals)
        return float(vals[0]), float(vals[1]), int(np.sum(vals < cfg["index_neg_tol"])), "", time.perf_counter() - t0
    except Exception as ex:
        return np.nan, np.nan, -1, f"index verification failed: {type(ex).__name__}: {ex}", time.perf_counter() - t0


def history_stats(hist):
    arr = np.asarray(hist, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    init, fin = float(arr[0]), float(arr[-1])
    rel = float(fin / init) if np.isfinite(init) and init > 0 and np.isfinite(fin) else np.nan
    return init, fin, rel


def initialize_standard_direction(u0, A, p, cfg):
    H0 = hess(u0, A, p)
    init_lambda, note = np.nan, ""
    t0 = time.perf_counter()
    try:
        vals, vecs = eigsh(H0, k=1, which="SA", tol=cfg["eigsh_tol"], maxiter=cfg["eigsh_maxiter"])
        v0, init_lambda = normalize_l2(vecs[:, 0]), float(vals[0])
    except Exception as ex:
        v0 = normalize_l2(u0.copy())
        note = f"ordinary eigsh failed; used L2-normalized u0 ({type(ex).__name__}: {ex})"
    return {"v0": v0, "init_lambda": init_lambda, "init_eig_time": float(time.perf_counter() - t0), "note": note}


def setup_h1_preconditioner_and_direction(u0, A, p, alpha, cfg):
    n, note = u0.size, ""
    t_setup = time.perf_counter()
    M = (A + alpha * sp.eye(n, format="csr")).tocsc()
    t_factor = time.perf_counter()
    factor = splu(M)
    factor_time = time.perf_counter() - t_factor

    def solve_M(b):
        return factor.solve(b)

    def M_dot(v):
        return M @ v

    init_lambda = np.nan
    t_eig = time.perf_counter()
    try:
        vals, vecs = eigsh(hess(u0, A, p), M=M, k=1, which="SA", tol=cfg["eigsh_tol"], maxiter=cfg["eigsh_maxiter"])
        v0, init_lambda = normalize_M(vecs[:, 0], M_dot), float(vals[0])
    except Exception as ex:
        v0 = normalize_M(u0.copy(), M_dot)
        note = f"generalized eigsh failed; used M-normalized u0 ({type(ex).__name__}: {ex})"

    return {
        "M": M,
        "solve_M": solve_M,
        "M_dot": M_dot,
        "v0": v0,
        "init_lambda": init_lambda,
        "note": note,
        "setup_time": float(time.perf_counter() - t_setup),
        "factor_time": float(factor_time),
        "init_eig_time": float(time.perf_counter() - t_eig),
    }


def run_standard_single_eta(u0, A, p, h, eta_x, tau_v, cfg, max_iter):
    u = u0.copy()
    grad_hist, E_hist = [], []
    t_setup = time.perf_counter()
    init = initialize_standard_direction(u0, A, p, cfg)
    v = init["v0"].copy()
    setup_time = time.perf_counter() - t_setup

    status = "max_iter"
    t_iter = time.perf_counter()
    for _ in range(max_iter):
        Au = A @ u
        g = Au - u**p
        ng_raw = float(np.linalg.norm(g))
        ng = residual_h(g, h)
        E = 0.5 * float(u @ Au) - (1.0 / (p + 1.0)) * float(np.sum(u ** (p + 1)))
        grad_hist.append(ng)
        E_hist.append(float(E))

        # Keep current A behavior: diverge checks before success.
        if (not np.isfinite(ng_raw)) or (not np.isfinite(ng)) or (not np.isfinite(E)) or (not np.all(np.isfinite(u))):
            status = "diverged"
            break
        if ng_raw > cfg["div_grad"] or abs(E) > cfg["div_E"]:
            status = "diverged"
            break
        if ng < cfg["tol"]:
            status = "success"
            break

        H = hess(u, A, p)
        for _j in range(cfg["J"]):
            Hv = H @ v
            v = normalize_l2(v - tau_v * (Hv - v * (v @ Hv)))
        u = u + eta_x * (-g + 2.0 * v * (v @ g))

    iter_time = time.perf_counter() - t_iter
    return {
        "method": "standard_HiSD",
        "status": status,
        "iter": len(grad_hist),
        "eta_x": float(eta_x),
        "tau_v": float(tau_v),
        "J": int(cfg["J"]),
        "max_iter_budget": int(max_iter),
        "u_final": u,
        "grad_norm_history": np.array(grad_hist, dtype=float),
        "energy_history": np.array(E_hist, dtype=float),
        "setup_time": float(setup_time),
        "iter_time": float(iter_time),
        "time_total": float(setup_time + iter_time),
        "factor_time": 0.0,
        "init_eig_time": float(init["init_eig_time"]),
        "init_lambda": float(init["init_lambda"]) if np.isfinite(init["init_lambda"]) else np.nan,
        "note": init["note"],
    }


def run_standard_fixed(u0, A, p, h, eta_x, tau_v, cfg, max_iter):
    return run_standard_single_eta(u0, A, p, h, eta_x, tau_v, cfg, max_iter)


def run_phisd_h1_single_eta(u0, A, p, h, eta_x, cfg, max_iter, setup, success_check_first):
    u = u0.copy()
    v = setup["v0"].copy()
    solve_M, M_dot = setup["solve_M"], setup["M_dot"]
    grad_hist, E_hist = [], []

    status = "max_iter"
    t_iter = time.perf_counter()
    for _ in range(max_iter):
        g = grad(u, A, p)
        ng_raw = float(np.linalg.norm(g))
        ng = residual_h(g, h)
        E = energy(u, A, p)
        grad_hist.append(ng)
        E_hist.append(float(E))

        if success_check_first and ng < cfg["tol"]:
            status = "success"
            break
        if (not np.isfinite(ng_raw)) or (not np.isfinite(ng)) or (not np.isfinite(E)) or (not np.all(np.isfinite(u))):
            status = "diverged"
            break
        if ng_raw > cfg["div_grad"] or abs(E) > cfg["div_E"]:
            status = "diverged"
            break
        if (not success_check_first) and ng < cfg["tol"]:
            status = "success"
            break

        H = hess(u, A, p)
        for _j in range(cfg["J"]):
            Hv = H @ v
            w = solve_M(Hv)
            v = normalize_M(v - cfg["h1_tau"] * (w - v * (v @ (M_dot(w)))), M_dot)
        u = u + eta_x * (-solve_M(g) + 2.0 * v * (v @ g))

    return {
        "status": status,
        "iter": len(grad_hist),
        "eta_x": float(eta_x),
        "tau_v": float(cfg["h1_tau"]),
        "J": int(cfg["J"]),
        "max_iter_budget": int(max_iter),
        "u_final": u,
        "grad_norm_history": np.array(grad_hist, dtype=float),
        "energy_history": np.array(E_hist, dtype=float),
        "iter_time": float(time.perf_counter() - t_iter),
    }


def run_h1_fixed(u0, A, p, h, alpha, cfg, success_check_first, catch_setup_error):
    try:
        setup = setup_h1_preconditioner_and_direction(u0, A, p, alpha, cfg)
    except Exception as ex:
        if not catch_setup_error:
            raise
        return {
            "status": "setup_failed",
            "iter": 0,
            "eta_x": np.nan,
            "tau_v": float(cfg["h1_tau"]),
            "J": int(cfg["J"]),
            "max_iter_budget": int(cfg["h1_max_iter"]),
            "u_final": u0.copy(),
            "grad_norm_history": np.array([], dtype=float),
            "energy_history": np.array([], dtype=float),
            "setup_time": 0.0,
            "iter_time": 0.0,
            "time_total": 0.0,
            "factor_time": 0.0,
            "init_eig_time": 0.0,
            "init_lambda": np.nan,
            "note": f"setup_failed: {type(ex).__name__}: {ex}",
        }

    run = run_phisd_h1_single_eta(u0, A, p, h, cfg["h1_eta"], cfg, cfg["h1_max_iter"], setup, success_check_first)
    run.update({
        "setup_time": float(setup["setup_time"]),
        "iter_time": float(run["iter_time"]),
        "time_total": float(setup["setup_time"] + run["iter_time"]),
        "factor_time": float(setup["factor_time"]),
        "init_eig_time": float(setup["init_eig_time"]),
        "init_lambda": float(setup["init_lambda"]) if np.isfinite(setup["init_lambda"]) else np.nan,
    })
    run["note"] = merge_notes(setup.get("note", ""))
    return run


def make_A_row(p, n, h, method, run, eig, note):
    init, fin, rel = history_stats(run["grad_norm_history"])
    l1, l2, idx, idx_t = eig
    return {
        "p": p,
        "N": A_N,
        "n": n,
        "h": float(h),
        "amp": float(A_AMP),
        "method": method,
        "eta_x": run["eta_x"],
        "tau_v": run["tau_v"],
        "J": run["J"],
        "max_iter_budget": run["max_iter_budget"],
        "status": run["status"],
        "iter": run["iter"],
        "initial_residual": init,
        "final_residual": fin,
        "relative_final_residual": rel,
        "lambda1": l1,
        "lambda2": l2,
        "index": idx,
        "init_lambda": run["init_lambda"],
        "setup_time": run["setup_time"],
        "iter_time": run["iter_time"],
        "time_total": run["time_total"],
        "factor_time": run["factor_time"],
        "init_eig_time": run["init_eig_time"],
        "index_verify_time": idx_t,
        "u_min": float(np.min(run["u_final"])),
        "u_max": float(np.max(run["u_final"])),
        "u_l2": float(np.linalg.norm(run["u_final"])),
        "note": note,
    }


def make_B_row(p, N, n, h, run, eig, note):
    init, fin, rel = history_stats(run["grad_norm_history"])
    l1, l2, idx, idx_t = eig
    return {
        "p": p,
        "N": N,
        "n": n,
        "h": float(h),
        "amp": float(B_AMP),
        "alpha": float(B_ALPHA),
        "method": "pHiSD_H1",
        "eta_x": run["eta_x"],
        "tau_v": run["tau_v"],
        "J": run["J"],
        "max_iter_budget": run["max_iter_budget"],
        "status": run["status"],
        "iter": run["iter"],
        "initial_grad_norm": init,
        "final_grad_norm": fin,
        "relative_final_grad_norm": rel,
        "lambda1": l1,
        "lambda2": l2,
        "index": idx,
        "init_lambda": run["init_lambda"],
        "setup_time": run["setup_time"],
        "iter_time": run["iter_time"],
        "time_total": run["time_total"],
        "factor_time": run["factor_time"],
        "init_eig_time": run["init_eig_time"],
        "index_verify_time": idx_t,
        "u_min": float(np.min(run["u_final"])),
        "u_max": float(np.max(run["u_final"])),
        "u_l2": float(np.linalg.norm(run["u_final"])),
        "note": note,
    }


def save_first300_convergence_figure_A(results, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    specs = [
        (3, "standard_HiSD", r"$p=3$, HiSD", "#1f77b4", "-", 1.5),
        (3, "pHiSD_H1", r"$p=3$, p-HiSD", "#d62728", "-", 1.5),
        (5, "pHiSD_H1", r"$p=5$, p-HiSD", "#ff7f0e", "-", 1.5),
    ]
    for p, method, label, color, ls, lw in specs:
        rec = results.get((p, method))
        if rec is None or len(rec["grad_norm_history"]) == 0:
            continue
        y = rec["grad_norm_history"]
        m = min(len(y), A_PLOT_FIRST_M + 1)
        ax.semilogy(np.arange(m), y[:m], color=color, linestyle=ls, linewidth=lw, label=label)

    ax.set_xlim(0, A_PLOT_FIRST_M)
    ax.set_ylim(1e-7, 1e1)
    ax.set_xlabel("Iteration $m$", fontsize=17)
    ax.set_ylabel(r"$\|\nabla E_h(U_m)\|_h$", fontsize=17)
    ax.set_title("2D Lane--Emden equation", fontsize=17)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=15, loc="center right", bbox_to_anchor=(1.0, 0.70))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment_A(out_dir):
    print("Running Experiment A: fixed-N convergence figure")
    print(f"  N = {A_N}, p_list = {A_P_LIST}, amp = {A_AMP}")
    print(f"  standard: eta = {A_STD_ETA_X}, tau = {A_STD_TAU_V}, J = {A_STD_J_INNER}, max_iter = {A_STD_MAX_ITER}")
    print(f"  p-HiSD-H1: eta = {A_H1_ETA_X}, tau = {A_H1_TAU_V}, J = {A_H1_J_INNER}, max_iter = {A_H1_MAX_ITER}")
    print(f"  tol = {A_TOL}")

    out_csv = out_dir / A_OUT_SUMMARY_CSV
    out_npz = out_dir / A_OUT_HISTORIES_NPZ

    pdf_dir = (Path(__file__).resolve().parent / ".." / ".." / "figures").resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = pdf_dir / A_OUT_FIRST300_PDF  

    rows, results = [], {}
    A_mat = build_laplacian_2d(A_N)
    u0, h = make_u0(A_N, A_AMP)
    n = A_N * A_N

    for p in A_P_LIST:
        std = run_standard_fixed(
            u0,
            A_mat,
            p,
            h,
            eta_x=A_STD_ETA_X,
            tau_v=A_STD_TAU_V,
            cfg=A_STD_CFG,
            max_iter=A_STD_MAX_ITER,
        )
        l1, l2, idx, idx_note, idx_t = verify_index(std["u_final"], A_mat, p, A_STD_CFG)
        note = merge_notes(std.get("note", ""), idx_note)
        if std["status"] == "max_iter":
            note = merge_notes(note, "not converged within iteration budget")
        if std["status"] == "success" and idx != 1:
            note = merge_notes(note, "wrong final index")
        std["residual_history"] = std["grad_norm_history"]
        std["final_residual"] = float(std["grad_norm_history"][-1]) if len(std["grad_norm_history"]) else np.nan
        rows.append(make_A_row(p, n, h, "standard_HiSD", std, (l1, l2, idx, idx_t), note))
        results[(p, "standard_HiSD")] = std

        h1 = run_h1_fixed(
            u0,
            A_mat,
            p,
            h,
            A_ALPHA,
            A_CFG,
            success_check_first=False,
            catch_setup_error=False,
        )
        l1, l2, idx, idx_note, idx_t = verify_index(h1["u_final"], A_mat, p, A_CFG)
        note = merge_notes(h1.get("note", ""), idx_note)
        if h1["status"] == "success" and idx != 1:
            note = merge_notes(note, "wrong final index")
        if h1["status"] == "max_iter":
            note = merge_notes(note, "not converged within iteration budget")
        h1["residual_history"] = h1["grad_norm_history"]
        h1["final_residual"] = float(h1["grad_norm_history"][-1]) if len(h1["grad_norm_history"]) else np.nan
        rows.append(make_A_row(p, n, h, "pHiSD_H1", h1, (l1, l2, idx, idx_t), note))
        results[(p, "pHiSD_H1")] = h1

    save_csv(rows, A_FIELDS, out_csv)
    save_first300_convergence_figure_A(results, out_pdf)

    payload = {}
    for p in A_P_LIST:
        for method in ["standard_HiSD", "pHiSD_H1"]:
            rec = results[(p, method)]
            tag = "standard" if method == "standard_HiSD" else "H1"
            payload[f"p{p}_{tag}_residual_history"] = rec["grad_norm_history"]
            payload[f"p{p}_{tag}_energy_history"] = rec["energy_history"]
            payload[f"p{p}_{tag}_eta_x"] = np.array([rec["eta_x"]], dtype=float)
            payload[f"p{p}_{tag}_status"] = np.array([rec["status"]], dtype="U64")
            payload[f"p{p}_{tag}_iter"] = np.array([rec["iter"]], dtype=int)
    np.savez(out_npz, **payload)

    for p in A_P_LIST:
        for method in ["standard_HiSD", "pHiSD_H1"]:
            row = next(r for r in rows if r["p"] == p and r["method"] == method)
            rel = row["relative_final_residual"]
            print(
                f"[A] p={p} method={method:<14} status={row['status']:<9} iter={row['iter']:<6d} "
                f"res_h={row['final_residual']:.3e} rel_grad={(f'{rel:.3e}' if np.isfinite(rel) else 'nan')} index={row['index']:<2d}"
            )

    return {"csv": out_csv, "pdf": out_pdf, "npz": out_npz}


def run_experiment_B(out_dir):
    print("\nRunning Experiment B: mesh-independence table")
    print(f"  P_LIST = {B_P_LIST}")
    print(f"  N_LIST = {B_N_LIST}")
    print(f"  AMP = {B_AMP}")
    print(f"  ALPHA = {B_ALPHA}")
    print(f"  TOL = {B_TOL}")
    print(f"  H1_ETA_X = {B_H1_ETA_X}")
    print(f"  H1_TAU_V = {B_H1_TAU_V}")
    print(f"  J_INNER = {B_H1_J_INNER}")
    print(f"  H1_MAX_ITER = {B_H1_MAX_ITER}")

    out_csv = out_dir / B_OUT_SUMMARY_CSV
    out_npz = out_dir / B_OUT_HISTORIES_NPZ

    rows, payload = [], {}
    for p in B_P_LIST:
        for N in B_N_LIST:
            A_mat = build_laplacian_2d(N)
            u0, h = make_u0(N, B_AMP)
            n = N * N

            run = run_h1_fixed(
                u0,
                A_mat,
                p,
                h,
                B_ALPHA,
                B_CFG,
                success_check_first=True,
                catch_setup_error=True,
            )
            l1, l2, idx, idx_note, idx_t = verify_index(run["u_final"], A_mat, p, B_CFG)
            note = merge_notes(run.get("note", ""), idx_note)
            if run["status"] == "success" and idx != 1:
                note = merge_notes(note, "wrong final index")
            row = make_B_row(p, N, n, h, run, (l1, l2, idx, idx_t), note)
            rows.append(row)

            print(
                f"[B] p={p} N={N} method=pHiSD_H1 status={row['status']} iter={row['iter']} "
                f"res_h={row['final_grad_norm']:.3e} index={row['index']} eta={row['eta_x']:.3g}"
            )

            tag = f"p{p}_N{N}"
            payload[f"{tag}_grad_norm_history"] = run["grad_norm_history"]
            payload[f"{tag}_energy_history"] = run["energy_history"]
            payload[f"{tag}_eta_x"] = np.array([run["eta_x"]], dtype=float)
            payload[f"{tag}_status"] = np.array([run["status"]], dtype="U64")
            payload[f"{tag}_iter"] = np.array([run["iter"]], dtype=int)
            payload[f"{tag}_lambda1"] = np.array([l1], dtype=float)
            payload[f"{tag}_lambda2"] = np.array([l2], dtype=float)
            payload[f"{tag}_index"] = np.array([idx], dtype=int)

    save_csv(rows, B_FIELDS, out_csv)
    np.savez(out_npz, **payload)
    return {"csv": out_csv, "npz": out_npz}


def main():
    set_paper_style()
    np.random.seed(RANDOM_SEED)
    out_dir = Path(__file__).resolve().parent
    outA = run_experiment_A(out_dir)
    outB = run_experiment_B(out_dir)
    print("\nSaved outputs:")
    print(outA["csv"])
    print(outA["pdf"])
    print(outA["npz"])
    print(outB["csv"])
    print(outB["npz"])


if __name__ == "__main__":
    main()
