# -*- coding: utf-8 -*-
import os, sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

# Avoid local copy.py shadowing stdlib copy (breaks sympy/mpmath import)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path: sys.path.remove(_THIS_DIR)
import copy as _stdlib_copy
sys.modules["copy"] = _stdlib_copy
if _THIS_DIR not in sys.path: sys.path.insert(0, _THIS_DIR)

import numpy as np
import sympy
from scipy.linalg import eigh

@dataclass(frozen=True)
class Config:
    alpha: float = 0.5
    a1: float = 0.49
    eps_M: float = 1e-2
    dt_x: float = 0.05
    dt_x_euc: float = 0.0001
    dt_v: float = 0.001
    steps_v: int = 5
    max_steps: int = 10000
    tol_grad: float = 1e-6
    domain_r: float = 5.0
    target_saddle: Tuple[float, float] = (0.06601892873348952, 0.18404094062214515)
    target_radius: float = 1e-2
    x_range_default: Tuple[float, float] = (-0.8, -0.5)
    y_range_default: Tuple[float, float] = (1.2, 1.5)
    x_range_sanity: Tuple[float, float] = (-0.68, -0.56)
    y_range_sanity: Tuple[float, float] = (1.2733333333, 1.3533333333)
    n_samples: int = 500
    seed: int = 42

CFG = Config()
_WORKER_MODEL = None
_WORKER_CFG: Config | None = None

class MullerBrownEnergy:
    def __init__(self):
        x, y = sympy.symbols("x y", real=True)
        A = [-200, -100, -170, 15]
        a = [-1, -1, -6.5, 0.7]
        b = [0, 0, 11, 0.6]
        c = [-10, -10, -6.5, 0.7]
        xb = [1, 0, -0.5, -1]
        yb = [0, 0.5, 1.5, 1]
        E = sum(A[i] * sympy.exp(a[i] * (x - xb[i]) ** 2 + b[i] * (x - xb[i]) * (y - yb[i]) + c[i] * (y - yb[i]) ** 2) for i in range(4))
        E += 500 * sympy.sin(x * y) * sympy.exp(-0.1 * (x + 0.5582) ** 2 - 0.1 * (y - 1.4417) ** 2)
        self.Ef = sympy.lambdify((x, y), E, "numpy")
        gx, gy = sympy.diff(E, x), sympy.diff(E, y)
        self.Gf = sympy.lambdify((x, y), sympy.Matrix([gx, gy]), "numpy")
        H = sympy.Matrix([[sympy.diff(gx, x), sympy.diff(gx, y)], [sympy.diff(gy, x), sympy.diff(gy, y)]])
        self.Hf = sympy.lambdify((x, y), H, "numpy")
    def energy(self, xy: np.ndarray) -> float:
        try:
            z = self.Ef(xy[0], xy[1])
            return float(z) if np.isfinite(z).all() else float(np.nan_to_num(z, nan=0.0, posinf=1e12, neginf=-1e12))
        except Exception:
            return 0.0
    def grad(self, xy: np.ndarray) -> np.ndarray:
        try:
            g = np.array(self.Gf(xy[0], xy[1]), dtype=float).reshape(-1)
            return g if np.isfinite(g).all() else np.full(2, np.nan, dtype=float)
        except Exception:
            return np.full(2, np.nan, dtype=float)
    def hess(self, xy: np.ndarray) -> np.ndarray:
        try:
            H = np.array(self.Hf(xy[0], xy[1]), dtype=float)
            return H if np.isfinite(H).all() else np.full((2, 2), np.nan, dtype=float)
        except Exception:
            return np.full((2, 2), np.nan, dtype=float)

def symmetrize(A: np.ndarray) -> np.ndarray: return 0.5 * (A + A.T)
def finite(a: np.ndarray | float) -> bool: return bool(np.isfinite(a).all())

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v.copy() if (not np.isfinite(n)) or n < 1e-16 else v / n

def normalize_M(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    val = float(v @ (M @ v))
    return normalize(v) if (not np.isfinite(val)) or val <= 1e-14 else v / np.sqrt(val)

def smallest_eigvec(H: np.ndarray, M: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
    if M is None:
        evals, evecs = np.linalg.eigh(H)
        return normalize(evecs[:, 0]), float(evals[0])
    try:
        evals, evecs = eigh(H, M, subset_by_index=[0, 0])
        return normalize_M(evecs[:, 0], M), float(evals[0])
    except Exception:
        evals, evecs = np.linalg.eigh(H)
        return normalize_M(evecs[:, 0], M), float(evals[0])

class InertialPreconditioner:
    def __init__(self, alpha: float, a1: float, eps: float):
        self.alpha, self.a1, self.eps, self.v_s = alpha, a1, eps, None
    def build_M(self, H: np.ndarray) -> np.ndarray:
        evals, evecs = np.linalg.eigh(H)
        v = normalize(evecs[:, 0])
        if self.v_s is not None:
            if np.dot(v, self.v_s) < 0: v = -v
            v = normalize((1 - self.alpha) * v + self.alpha * self.v_s)
        self.v_s = v
        mu1 = self.a1 * abs(float(evals[0])) + self.eps
        mu2 = (1 - self.a1) * abs(float(evals[1])) + self.eps
        return (mu1 - mu2) * np.outer(v, v) + mu2 * np.eye(2)

def evolve_v_precond(v: np.ndarray, H: np.ndarray, M: np.ndarray, steps: int, dt: float) -> np.ndarray:
    v = normalize_M(v.copy(), M)
    I = np.eye(len(v))
    for _ in range(steps):
        Hv = H @ v
        try: Minv_Hv = np.linalg.solve(M, Hv)
        except Exception: Minv_Hv = Hv.copy()
        force = -(I - np.outer(v, v @ M)) @ Minv_Hv
        fn = np.linalg.norm(force)
        if (not np.isfinite(fn)) or fn < 1e-10: break
        v = normalize_M(v + dt * force, M)
        if not finite(v): break
    return v

def run_solver(model: MullerBrownEnergy, x0: np.ndarray, cfg: Config, method: str) -> Tuple[np.ndarray, str]:
    x, path, status = x0.copy(), [x0.copy()], "Timeout"
    precon = InertialPreconditioner(cfg.alpha, cfg.a1, cfg.eps_M) if method == "subspace" else None
    H = model.hess(x)
    if not finite(H): return np.array(path), "Numerical/Diverged"
    M = precon.build_M(H) if precon is not None else None
    if M is not None and not finite(M): return np.array(path), "Numerical/Diverged"
    v, _ = smallest_eigvec(H, M)

    for _ in range(cfg.max_steps):
        g = model.grad(x)
        if not finite(g): status = "Numerical/Diverged"; break
        gnorm = np.linalg.norm(g)
        if not np.isfinite(gnorm): status = "Numerical/Diverged"; break
        if gnorm < cfg.tol_grad: status = "Converged"; break
        H = model.hess(x)
        if not finite(H): status = "Numerical/Diverged"; break

        if method == "subspace":
            M = precon.build_M(H)
            if not finite(M): status = "Numerical/Diverged"; break
            v = evolve_v_precond(v, H, M, cfg.steps_v, cfg.dt_v)
            if not finite(v): status = "Numerical/Diverged"; break
            try: Minv_g = np.linalg.solve(M, g)
            except Exception: Minv_g = g.copy()
            if not finite(Minv_g): status = "Numerical/Diverged"; break
            step = cfg.dt_x * (-(Minv_g - 2.0 * v * float(v @ g)))
            if not finite(step): status = "Numerical/Diverged"; break
        else:
            I = np.eye(len(v))
            for _ in range(cfg.steps_v):
                dv = -(I - np.outer(v, v)) @ (H @ v)
                if not finite(dv): status = "Numerical/Diverged"; break
                v = normalize(v + cfg.dt_v * dv)
                if not finite(v): status = "Numerical/Diverged"; break
            if status == "Numerical/Diverged": break
            step = cfg.dt_x_euc * (-(g - 2.0 * v * float(v @ g)))
            if not finite(step): status = "Numerical/Diverged"; break

        x = x + step
        if not finite(x): status = "Numerical/Diverged"; break
        path.append(x.copy())
        xn = np.linalg.norm(x)
        if (not np.isfinite(xn)) or (xn > cfg.domain_r):
            status = "Out of Bounds" if np.isfinite(xn) else "Numerical/Diverged"
            break
    return np.array(path), status

def classify_endpoint(model: MullerBrownEnergy, path: np.ndarray, solver_status: str, cfg: Config) -> Tuple[bool, str]:
    if solver_status in {"Numerical/Diverged", "Out of Bounds", "Timeout"}: return False, solver_status
    x = np.array(path[-1], dtype=float)
    if not finite(x): return False, "Numerical/Diverged"
    xn = np.linalg.norm(x)
    if (not np.isfinite(xn)) or (xn > cfg.domain_r): return False, "Out of Bounds"
    g, H = model.grad(x), model.hess(x)
    if (not finite(g)) or (not finite(H)): return False, "Numerical/Diverged"
    gnorm = np.linalg.norm(g)
    if not np.isfinite(gnorm): return False, "Numerical/Diverged"
    try: eigvals = np.linalg.eigvalsh(symmetrize(H))
    except Exception: return False, "Numerical/Diverged"
    is_1_saddle = (eigvals[0] < -1e-3) and (eigvals[1] > 1e-3)
    if is_1_saddle:
        d = np.linalg.norm(x - np.array(cfg.target_saddle, dtype=float))
        return (True, "Target 1-Saddle") if np.isfinite(d) and d <= cfg.target_radius else (False, "Other 1-Saddle")
    return (False, "Stationary but Not 1-Saddle") if (solver_status == "Converged" or gnorm < cfg.tol_grad) else (False, "Numerical/Diverged")

def _init_worker(cfg: Config):
    global _WORKER_MODEL, _WORKER_CFG
    _WORKER_MODEL, _WORKER_CFG = MullerBrownEnergy(), cfg

def _run_single(task: Tuple[str, np.ndarray]) -> Dict[str, str | int]:
    method, x0 = task
    path, st = run_solver(_WORKER_MODEL, np.array(x0, dtype=float), _WORKER_CFG, method)
    ok, reason = classify_endpoint(_WORKER_MODEL, path, st, _WORKER_CFG)
    return {"success": int(ok), "reason": reason}

def run_batch(method: str, init_points: np.ndarray, cfg: Config, n_workers: int) -> List[Dict[str, str | int]]:
    tasks = [(method, p) for p in init_points]
    res = []
    print(f"[{method}] Launching {len(tasks)} tasks with {n_workers} workers...")
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single, tasks), 1):
            res.append(r)
            if i % 10 == 0 or i == len(tasks): print(f"[{method}] Progress: {i}/{len(tasks)}")
    return res

def summarize(results: List[Dict[str, str | int]]) -> Dict[str, float | int]:
    n = len(results)
    names = ["Target 1-Saddle", "Other 1-Saddle", "Out of Bounds", "Timeout", "Numerical/Diverged", "Stationary but Not 1-Saddle"]
    cnt = {k: sum(r["reason"] == k for r in results) for k in names}
    return {
        "total": n, "success": cnt["Target 1-Saddle"], "rate": 100.0 * cnt["Target 1-Saddle"] / n if n else 0.0,
        "target": cnt["Target 1-Saddle"], "other": cnt["Other 1-Saddle"], "oob": cnt["Out of Bounds"],
        "timeout": cnt["Timeout"], "num": cnt["Numerical/Diverged"], "stationary": cnt["Stationary but Not 1-Saddle"],
    }

def fmt_count(count: int, total: int) -> str:
    rate = 100.0 * count / total if total else 0.0
    return f"{count}/{total} ({rate:.1f}\\%)"

def print_latex_endpoint_table(s_orig: Dict[str, float | int], s_sub: Dict[str, float | int]) -> None:
    rows = [
        ("HiSD", s_orig),
        ("Subspace-inertial p-HiSD", s_sub),
    ]

    print("\n=== Endpoint statistics for LaTeX ===")
    print("Method & Target 1-Saddle & Other 1-Saddle & Diverged/Failed \\\\")
    print("\\hline")

    for name, s in rows:
        total = int(s["total"])
        target = int(s["target"])
        other = int(s["other"])
        failed = total - target - other
        print(
            f"{name} & "
            f"{fmt_count(target, total)} & "
            f"{fmt_count(other, total)} & "
            f"{fmt_count(failed, total)} \\\\"
        )

    print("\n=== Plain CSV-style summary ===")
    print("Method,Target 1-Saddle,Other 1-Saddle,Diverged/Failed")
    for name, s in rows:
        total = int(s["total"])
        target = int(s["target"])
        other = int(s["other"])
        failed = total - target - other
        print(
            f"{name},"
            f"{target}/{total} ({100.0 * target / total if total else 0.0:.1f}%),"
            f"{other}/{total} ({100.0 * other / total if total else 0.0:.1f}%),"
            f"{failed}/{total} ({100.0 * failed / total if total else 0.0:.1f}%)"
        )

def choose_ranges(cfg: Config, sanity: bool) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (cfg.x_range_sanity, cfg.y_range_sanity) if sanity else (cfg.x_range_default, cfg.y_range_default)

def main():
    cfg = CFG
    sanity = os.environ.get("SANITY_RECT", "0") == "1"
    xr, yr = choose_ranges(cfg, sanity)
    print(f"Sampling region: X in {xr}, Y in {yr}, N={cfg.n_samples}, seed={cfg.seed}")
    print(f"Sanity sub-rectangle mode: {'ON' if sanity else 'OFF'}")
    print(f"Target 1-Saddle = {list(cfg.target_saddle)}, threshold = {cfg.target_radius}, DOMAIN_R = {cfg.domain_r}")
    np.random.seed(cfg.seed)
    init_points = np.column_stack([np.random.uniform(xr[0], xr[1], size=cfg.n_samples), np.random.uniform(yr[0], yr[1], size=cfg.n_samples)])
    n_workers = min(cpu_count(), 8)
    r_orig = run_batch("original", init_points, cfg, n_workers)
    r_sub = run_batch("subspace", init_points, cfg, n_workers)
    s_orig, s_sub = summarize(r_orig), summarize(r_sub)
    print("\n=== Summary ===")
    print(f"Original HiSD (Target 1-Saddle): {s_orig['success']}/{s_orig['total']} ({s_orig['rate']:.1f}%)")
    print(f"  Breakdown -> Other 1-Saddle: {s_orig['other']}, Stationary but Not 1-Saddle: {s_orig['stationary']}, Timeout: {s_orig['timeout']}, Out of Bounds: {s_orig['oob']}, Numerical/Diverged: {s_orig['num']}")
    print(f"Subspace p-HiSD (Target 1-Saddle): {s_sub['success']}/{s_sub['total']} ({s_sub['rate']:.1f}%)")
    print(f"  Breakdown -> Other 1-Saddle: {s_sub['other']}, Stationary but Not 1-Saddle: {s_sub['stationary']}, Timeout: {s_sub['timeout']}, Out of Bounds: {s_sub['oob']}, Numerical/Diverged: {s_sub['num']}")
    print_latex_endpoint_table(s_orig, s_sub)

if __name__ == "__main__":
    main()
