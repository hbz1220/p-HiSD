#!/usr/bin/env python3
# Section 7.2.2: paired step-size sweep on the modified Muller potential.
# Both methods use the same 512 starts at each of 41 step sizes.
# Save Figure 3, endpoint counts and resumable records to outputs/7.2/7.2.2/.

from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
from functools import lru_cache
import inspect
import importlib.util
import json
import csv
import math
import multiprocessing
from pathlib import Path
import platform
import signal
import sys
import tempfile
import textwrap
from typing import Tuple


def _ensure_dependencies():
    packages = ("numpy", "scipy", "sympy", "matplotlib")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if not missing:
        return
    script = Path(__file__).resolve()
    if os.environ.get("SECTION722_REEXEC") != "1":
        for parent in (script.parent, *script.parent.parents):
            for name in (".shared_venv", ".venv"):
                candidate = parent / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                if candidate.is_file():
                    os.environ["SECTION722_REEXEC"] = "1"
                    print(f"Using existing environment: {candidate}", flush=True)
                    os.execv(str(candidate), [str(candidate), str(script), *sys.argv[1:]])
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing)
                     + ". Select an interpreter with these packages installed.")


if __name__ == "__main__":
    _ensure_dependencies()

import numpy as np
import scipy
from scipy.linalg import eigh
import sympy

@dataclass(frozen=True)
class Config:
    alpha: float = 0.5
    a1: float = 0.49
    eps_M: float = 1e-2
    dt_x: float = 0.05
    dt_x_euc: float = 0.0001
    dt_v: float = 0.001
    steps_v: int = 5
    max_steps: int = 400000
    tol_grad: float = 1e-6
    domain_r: float = 5.0
    x_range_default: Tuple[float, float] = (-0.8, -0.5)
    y_range_default: Tuple[float, float] = (1.2, 1.5)
    x_range_sanity: Tuple[float, float] = (-0.68, -0.56)
    y_range_sanity: Tuple[float, float] = (1.2733333333, 1.3533333333)
    n_samples: int = 500
    seed: int = 42


# Differentiate the potential symbolically, then evaluate in float64.
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


# Align and blend the unstable eigenvector before assigning positive metric weights.
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


def load_source():
    return sys.modules[__name__]


class RawModel:
    """Source symbolic derivative lambdas, float64; no masked nonfinite energy."""
    def __init__(self):
        self.source_model = load_source().MullerBrownEnergy()
    def energy(self, x): return float(self.source_model.Ef(*x))
    def grad(self, x): return np.array(self.source_model.Gf(*x), dtype=float).reshape(2)
    def hess(self, x): return np.array(self.source_model.Hf(*x), dtype=float)


def raw_model():
    global _MODEL
    if _MODEL is None: _MODEL = RawModel()
    return _MODEL


# Success requires a fresh residual and one negative, one positive Hessian eigenvalue.
# Unresolved signs, wrong indices and exhausted budgets remain distinct outcomes.
def certify(x, reason='budget_exhausted', model=None):
    """Certify ONLY the supplied raw state, independently of previous residual."""
    model = raw_model() if model is None else model
    x = np.asarray(x, dtype=float)
    r = dict(grad_norm=None, lambda1=None, lambda2=None,
             terminal_category='numerical_breakdown', certification_reason='nonfinite_raw_state')
    if not np.isfinite(x).all(): return r
    try:
        with np.errstate(all='ignore'):
            g = model.grad(x); H = model.hess(x); n = float(np.linalg.norm(g))
            if not np.isfinite(g).all() or not np.isfinite(H).all() or not np.isfinite(n):
                r['certification_reason'] = 'nonfinite_derivatives_or_residual'; return r
            ev = np.linalg.eigvalsh(.5*(H+H.T))
            if not np.isfinite(ev).all():
                r['certification_reason'] = 'nonfinite_spectrum'; return r
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as e:
        r['certification_reason'] = type(e).__name__+': '+str(e); return r
    r.update(grad_norm=n, lambda1=float(ev[0]), lambda2=float(ev[1]), certification_reason='raw_state_verified')
    if n < 1e-6:
        if ev[0] < -1e-6 and ev[1] > 1e-6: c = 'success_index1'
        elif ev[0] > 1e-6: c = 'wrong_index_0'
        elif ev[1] < -1e-6: c = 'wrong_index_2'
        else: c = 'index_unresolved'
    else: c = reason
    r['terminal_category'] = c
    return r


# Fixed paper sweep: these settings and the grid below determine the experiment.
CFG = Config(eps_M=12.5, max_steps=2_000_000, dt_v=0.002)
_MODEL = None
CANCEL = False
NMAX = 2_000_000
N_SAMPLES = 512
J = 5
GUARD = 5.0
TAU = 2e-3
ETAS = tuple(float(x) for x in np.geomspace(1e-5, 1e-1, 41))
ETA0 = {"HiSD": 1e-5, "SI-pHiSD": 1e-5}
SAMPLING = "uniform_grid_32x16_centers"
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2] if _HERE.name == "7.2.2" and _HERE.parent.parent.name == "code" else _HERE
DEFAULT_OUTPUT = _REPO / "outputs/7.2/7.2.2"
CATEGORIES = ("success_index1", "budget_exhausted", "wrong_index_0", "wrong_index_2",
              "index_unresolved", "state_safeguard", "numerical_breakdown")
ENDPOINTS = {"S_H": (-2.6280457712730834, 1.7869725684854476),
             "S_L": (.06601892754199913, .18404094032481952)}
FLOAT_FIELDS = ("eta", "tau", "x0", "y0", "raw_x", "raw_y", "grad_norm", "lambda1", "lambda2")
INT_FIELDS = ("budget", "start_id", "n_outer_updates")
RAW_FIELDS = ["method", "eta", "tau", "budget", "start_id", "x0", "y0", "execution_status",
              "raw_x", "raw_y", "grad_norm", "lambda1", "lambda2", "n_outer_updates",
              "terminal_category", "endpoint", "stop_reason", "failure_reason", "certification_reason", "source"]
MAIN_FIELDS = ["method", "eta", "tau", "N", "sampling", "budget", "N_H", "N_L", "N_other",
               "success", "unsuccessful", "success_percent", "high_hit_percent", "low_hit_percent",
               *CATEGORIES[1:]]


def _dot(a, b):
    return (a[:, None, :] @ b[:, :, None])[:, 0, 0]


def _mv(a, v):
    return (a @ v[:, :, None])[:, :, 0]


def _norm(v):
    return np.sqrt(_dot(v, v))


def _finite(v):
    return np.isfinite(v).reshape(len(v), -1).all(axis=1)


def _normalize(v):
    out = v.copy()
    norm = _norm(v)
    use = np.isfinite(norm) & (norm >= 1e-16)
    out[use] /= norm[use, None]
    return out


def _normalize_m(v, metric):
    val = _dot(v, _mv(metric, v))
    use = np.isfinite(val) & (val > 1e-14)
    out = v.copy()
    out[use] /= np.sqrt(val[use, None])
    if (~use).any():
        out[~use] = _normalize(v[~use])
    return out


@lru_cache(maxsize=8)
def _scalar_rounding_lambda(function):
    """Preserve NumPy scalar power rounding in the array derivative lambda.

    ``np.float64(x) ** 2`` and ``array_x ** 2`` need not round identically:
    the array operator may replace a power by multiplication. In this sensitive
    dynamics that one-bit difference can change the eventual endpoint. A call
    to ``np.float_power`` keeps the scalar power semantics without evaluating
    trajectories one at a time. All other operations and their order are kept.
    """
    class ScalarPower(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Pow):
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="_batch_scalar_power", ctx=ast.Load()),
                    args=[node.left, node.right], keywords=[]), node)
            return node

    tree = ScalarPower().visit(ast.parse(textwrap.dedent(inspect.getsource(function))))
    ast.fix_missing_locations(tree)
    namespace = function.__globals__.copy()
    namespace["_batch_scalar_power"] = np.float_power
    exec(compile(tree, "<batched_derivative_scalar_power>", "exec"), namespace)
    return namespace[function.__name__]


def derivatives(solver, points, kind):
    """Evaluate the existing symbolic derivative lambda on all supplied rows.

    Individual evaluation is a rare exception fallback only. A bad row is
    returned as NaN rather than preventing unrelated rows from progressing.
    """
    points = np.asarray(points, dtype=float)
    shape = (len(points), 2) if kind == "grad" else (len(points), 2, 2)
    if kind not in ("grad", "hess"):
        raise ValueError("kind must be grad or hess")
    if not len(points):
        return np.empty(shape, dtype=float)
    model = solver.raw_model()
    source_function = model.source_model.Gf if kind == "grad" else model.source_model.Hf
    func = _scalar_rounding_lambda(source_function)
    try:
        value = np.asarray(func(points[:, 0], points[:, 1]), dtype=float)
        if kind == "grad":
            return np.moveaxis(value[:, 0, :], -1, 0).copy()
        return np.moveaxis(value, -1, 0).copy()
    except (FloatingPointError, OverflowError, ValueError, TypeError):
        out = np.full(shape, np.nan)
        scalar = model.grad if kind == "grad" else model.hess
        for row, point in enumerate(points):
            try:
                out[row] = scalar(point)
            except (FloatingPointError, OverflowError, ValueError, TypeError):
                pass
        return out


def _eigh(matrices):
    """Keep a rare failed eigensolve local to its trajectory."""
    try:
        values, vectors = np.linalg.eigh(matrices)
        return values, vectors, {}
    except np.linalg.LinAlgError:
        values = np.full((len(matrices), 2), np.nan)
        vectors = np.full((len(matrices), 2, 2), np.nan)
        errors = {}
        for row, matrix in enumerate(matrices):
            try:
                values[row], vectors[row] = np.linalg.eigh(matrix)
            except np.linalg.LinAlgError as exc:
                errors[row] = type(exc).__name__ + ": " + str(exc)
        return values, vectors, errors


def _solve(matrices, rhs):
    """The reference solver substitutes rhs when a matrix solve fails."""
    try:
        return np.linalg.solve(matrices, rhs[:, :, None])[:, :, 0], np.zeros(len(rhs), dtype=bool)
    except np.linalg.LinAlgError:
        out = rhs.copy()
        fallback = np.zeros(len(rhs), dtype=bool)
        for row, (matrix, vector) in enumerate(zip(matrices, rhs)):
            try:
                out[row] = np.linalg.solve(matrix, vector)
            except np.linalg.LinAlgError:
                fallback[row] = True
        return out, fallback


def run_batch(solver, points, method, eta, tau, budget,
              on_complete=None, stop_after=None, trace=False, on_progress=None):
    """Run independent trajectories together, preserving scalar stopping and update order."""
    if method not in solver.ETA0 or eta <= 0 or tau <= 0 or budget < 0:
        raise ValueError("invalid configuration")
    x = np.asarray(points, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("points must have shape (batch_size, 2)")
    size = len(x)
    if not size:
        return []
    v = np.full_like(x, np.nan)
    metric = np.full((size, 2, 2), np.nan)
    history = np.full_like(x, np.nan)
    has_v = np.zeros(size, dtype=bool)
    has_metric = np.zeros(size, dtype=bool)
    has_history = np.zeros(size, dtype=bool)
    last_finite = x.copy()
    active = np.ones(size, dtype=bool)
    results = [None] * size
    traces = [[] for _ in range(size)] if trace else None
    names = ("n_outer_updates", "n_grad_alg", "n_hess_assembly_alg", "n_M_setup",
             "n_M_update", "n_eigh_metric", "n_eigh_frame_init", "n_eigh_frame_inner",
             "n_grad_verify", "n_hess_assembly_verify", "n_state_M_solve_calls",
             "n_state_M_solve_fallback")
    counts = {name: np.zeros(size, dtype=np.int64) for name in names}
    model = solver.raw_model()
    eye = np.eye(2)
    preconditioned = method == "SI-pHiSD"
    progress_step = 0

    def remember(indices):
        if not trace:
            return
        for row in indices:
            traces[row].append(dict(update=int(counts["n_outer_updates"][row]), x=x[row].copy(),
                v=v[row].copy() if has_v[row] else None,
                M=metric[row].copy() if has_metric[row] else None,
                history=history[row].copy() if has_history[row] else None))

    def finish(indices, reason, detail="", execution="completed"):
        for row in np.asarray(indices, dtype=int):
            if not active[row]:
                continue
            if execution == "completed":
                cert = solver.certify(x[row], reason if reason != "gradient_tolerance" else "index_unresolved", model)
                counts["n_grad_verify"][row] = int(np.isfinite(x[row]).all())
                counts["n_hess_assembly_verify"][row] = int(np.isfinite(x[row]).all())
            else:
                cert = dict(grad_norm=None, lambda1=None, lambda2=None, terminal_category=None,
                            certification_reason="not_certified_interrupted")
            result = dict(execution_status=execution, raw_x=float(x[row, 0]), raw_y=float(x[row, 1]),
                last_finite_x=float(last_finite[row, 0]), last_finite_y=float(last_finite[row, 1]),
                stop_reason=reason, failure_reason=detail, **cert,
                **{name: int(value[row]) for name, value in counts.items()},
                n_Hv_alg=solver.J * int(counts["n_outer_updates"][row])
                    if method == "HiSD" and reason != "numerical_breakdown" else "unknown",
                n_inner_M_solve_calls="unknown", n_frame_normalizations="unknown",
                v=v[row].tolist() if has_v[row] else None,
                M=metric[row].tolist() if has_metric[row] else None,
                history=history[row].tolist() if has_history[row] else None)
            if trace:
                result["trace"] = traces[row]
            results[row] = result
            active[row] = False
            if execution == "completed" and on_complete is not None:
                on_complete(int(row), result)

    def bad(indices, mask, message):
        finish(indices[mask], "numerical_breakdown", "FloatingPointError: " + message)
        return ~mask

    with np.errstate(all="ignore"):
        # Initialize each trajectory with a scalar generalized eigensolve.
        for row in range(size):
            try:
                counts["n_hess_assembly_alg"][row] += 1
                hessian = model.hess(x[row])
                if not solver.finite(hessian):
                    raise FloatingPointError("initial_hessian_nonfinite")
                initial_metric = None
                if preconditioned:
                    pre = solver.InertialPreconditioner(solver.CFG.alpha, solver.CFG.a1, solver.CFG.eps_M)
                    initial_metric = pre.build_M(hessian)
                    metric[row] = initial_metric
                    has_metric[row] = True
                    if pre.v_s is not None:
                        history[row] = pre.v_s
                        has_history[row] = True
                    counts["n_M_setup"][row] += 1
                    counts["n_eigh_metric"][row] += 1
                    if not solver.finite(initial_metric):
                        raise FloatingPointError("initial_metric_nonfinite")
                v[row], _ = solver.smallest_eigvec(hessian, initial_metric)
                has_v[row] = True
                counts["n_eigh_frame_init"][row] += 1
                remember([row])
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                finish([row], "numerical_breakdown", type(exc).__name__ + ": " + str(exc))

        # Each trajectory stops independently; all outcomes remain in the denominator.
        while active.any():
            ids = np.flatnonzero(active)
            k = counts["n_outer_updates"][ids]
            interrupt = np.zeros(len(ids), dtype=bool)
            if stop_after is not None:
                interrupt |= (k >= stop_after) & (k < budget)
            if getattr(solver, "CANCEL", False):
                interrupt |= k % 50 == 0
            finish(ids[interrupt], "pending_restart_from_x0", execution="interrupted")
            ids = ids[~interrupt]
            if not len(ids):
                continue

            # Check the current residual before the frame sweeps and state update.
            counts["n_grad_alg"][ids] += 1
            grad = derivatives(solver, x[ids], "grad")
            keep = bad(ids, ~_finite(grad), "current_gradient_nonfinite")
            ids, grad = ids[keep], grad[keep]
            if not len(ids):
                continue
            norm = _norm(grad)
            keep = bad(ids, ~np.isfinite(norm), "current_residual_nonfinite")
            ids, grad, norm = ids[keep], grad[keep], norm[keep]
            finish(ids[norm < 1e-6], "gradient_tolerance")
            keep = norm >= 1e-6
            ids, grad = ids[keep], grad[keep]
            exhausted = counts["n_outer_updates"][ids] >= budget
            finish(ids[exhausted], "budget_exhausted")
            ids, grad = ids[~exhausted], grad[~exhausted]
            if not len(ids):
                continue

            counts["n_hess_assembly_alg"][ids] += 1
            hessian = derivatives(solver, x[ids], "hess")
            keep = bad(ids, ~_finite(hessian), "current_hessian_nonfinite")
            ids, grad, hessian = ids[keep], grad[keep], hessian[keep]
            if not len(ids):
                continue
            if preconditioned:
                eigenvalues, eigenvectors, errors = _eigh(hessian)
                if errors:
                    for local, detail in errors.items():
                        finish([ids[local]], "numerical_breakdown", detail)
                    keep = np.array([local not in errors for local in range(len(ids))])
                    ids, grad, hessian = ids[keep], grad[keep], hessian[keep]
                    eigenvalues, eigenvectors = eigenvalues[keep], eigenvectors[keep]
                if not len(ids):
                    continue
                direction = _normalize(eigenvectors[:, :, 0])
                direction[_dot(direction, history[ids]) < 0] *= -1
                direction = _normalize((1 - solver.CFG.alpha) * direction + solver.CFG.alpha * history[ids])
                history[ids] = direction
                has_history[ids] = True
                mu1 = solver.CFG.a1 * np.abs(eigenvalues[:, 0]) + solver.CFG.eps_M
                mu2 = (1 - solver.CFG.a1) * np.abs(eigenvalues[:, 1]) + solver.CFG.eps_M
                metric[ids] = ((mu1 - mu2)[:, None, None] * (direction[:, :, None] * direction[:, None, :])
                               + mu2[:, None, None] * eye)
                has_metric[ids] = True
                counts["n_M_update"][ids] += 1
                counts["n_eigh_metric"][ids] += 1
                keep = bad(ids, ~_finite(metric[ids]), "metric_nonfinite")
                ids, grad, hessian = ids[keep], grad[keep], hessian[keep]
                if not len(ids):
                    continue
                v[ids] = _normalize_m(v[ids], metric[ids])
                inner_active = np.ones(len(ids), dtype=bool)
                for _ in range(solver.J):
                    local = np.flatnonzero(inner_active)
                    if not len(local):
                        break
                    rows = ids[local]
                    current_v, current_m = v[rows], metric[rows]
                    hv = _mv(hessian[local], current_v)
                    minv_hv, _ = _solve(current_m, hv)
                    vm = (current_v[:, None, :] @ current_m)[:, 0, :]
                    force = _mv(-(eye - current_v[:, :, None] * vm[:, None, :]), minv_hv)
                    finite_force = np.isfinite(_norm(force))
                    inner_active[local[~finite_force]] = False
                    rows = rows[finite_force]
                    if len(rows):
                        v[rows] = _normalize_m(v[rows] + tau * force[finite_force], metric[rows])
                        inner_active[local[finite_force][~_finite(v[rows])]] = False
                keep = bad(ids, ~_finite(v[ids]), "frame_nonfinite")
                ids, grad = ids[keep], grad[keep]
                if not len(ids):
                    continue
                counts["n_state_M_solve_calls"][ids] += 1
                mg, fallback = _solve(metric[ids], grad)
                counts["n_state_M_solve_fallback"][ids] += fallback
                keep = bad(ids, ~_finite(mg), "inverse_metric_gradient_nonfinite")
                ids, grad, mg = ids[keep], grad[keep], mg[keep]
                if not len(ids):
                    continue
                step = eta * (-(mg - 2.0 * v[ids] * _dot(v[ids], grad)[:, None]))
            else:
                for _ in range(solver.J):
                    current = v[ids]
                    force = _mv(-(eye - current[:, :, None] * current[:, None, :]), _mv(hessian, current))
                    keep = bad(ids, ~_finite(force), "frame_force_nonfinite")
                    ids, grad, hessian, force = ids[keep], grad[keep], hessian[keep], force[keep]
                    if not len(ids):
                        break
                    v[ids] = _normalize(v[ids] + tau * force)
                    keep = bad(ids, ~_finite(v[ids]), "frame_nonfinite")
                    ids, grad, hessian = ids[keep], grad[keep], hessian[keep]
                    if not len(ids):
                        break
                if not len(ids):
                    continue
                step = eta * (-(grad - 2.0 * v[ids] * _dot(v[ids], grad)[:, None]))

            keep = bad(ids, ~_finite(step), "state_step_nonfinite")
            ids, step = ids[keep], step[keep]
            if not len(ids):
                continue
            x[ids] = x[ids] + step
            counts["n_outer_updates"][ids] += 1
            remember(ids)
            current_step = int(counts["n_outer_updates"][ids].max())
            if on_progress is not None and current_step >= progress_step + 10000:
                on_progress(current_step, int(active.sum()))
                progress_step = current_step
            keep = bad(ids, ~_finite(x[ids]), "updated_raw_state_nonfinite")
            ids = ids[keep]
            if not len(ids):
                continue
            last_finite[ids] = x[ids]
            norm = _norm(x[ids])
            keep = bad(ids, ~np.isfinite(norm), "state_norm_nonfinite")
            ids, norm = ids[keep], norm[keep]
            finish(ids[norm > solver.GUARD], "state_safeguard", "source_norm2_gt_5")
    return results


def configs():
    return [(method, eta) for eta in ETAS for method in ETA0]


# Deterministic cell centers in the prescribed rectangle, shared by both methods.
def initial_points():
    xs = -.8 + (np.arange(32) + .5) * (.3 / 32)
    ys = 1.2 + (np.arange(16) + .5) * (.3 / 16)
    return np.array([(x, y) for y in ys for x in xs])


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def endpoint_label(row):
    if row["terminal_category"] != "success_index1":
        return "U"
    hits = [label for label, point in ENDPOINTS.items()
            if math.hypot(row["raw_x"]-point[0], row["raw_y"]-point[1]) < 1e-5]
    if len(hits) > 1:
        raise ValueError("Ambiguous endpoint identity")
    return hits[0] if hits else "S_other"


def record_key(row):
    return row["method"], row["eta"], row["start_id"]


def validate_record(row):
    if (row["method"], row["eta"]) not in configs() or row["tau"] != TAU or row["budget"] != NMAX:
        raise ValueError("Raw record parameters do not match this sweep")
    i = row["start_id"]
    if not 0 <= i < N_SAMPLES or (row["x0"], row["y0"]) != tuple(initial_points()[i]):
        raise ValueError("Raw record initial point differs from the paired grid")
    if row["execution_status"] != "completed" or not 0 <= row["n_outer_updates"] <= NMAX:
        raise ValueError("Only complete trajectories within the budget may be recorded")
    if row["terminal_category"] not in CATEGORIES:
        raise ValueError("Unknown terminal category")
    finite_cert = all(isinstance(row[k], (int, float)) and math.isfinite(row[k])
                      for k in ("raw_x", "raw_y", "grad_norm", "lambda1", "lambda2"))
    success = (finite_cert and row["grad_norm"] < 1e-6
               and row["lambda1"] < -1e-6 and row["lambda2"] > 1e-6)
    if success != (row["terminal_category"] == "success_index1"):
        raise ValueError("Success differs from the raw gradient/index certification")
    if row["endpoint"] != endpoint_label(row):
        raise ValueError("Saved endpoint identity differs from raw coordinates")


def prepare_record(row, source):
    result = {k: row.get(k, "") for k in RAW_FIELDS}
    result.update(endpoint=endpoint_label(row), source=source)
    for k, value in result.items():
        if isinstance(value, str):
            result[k] = value.replace("\r", " ").replace("\n", " ")
    validate_record(result)
    return result


def read_records(path):
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        if end == 0:
            raise ValueError("Incomplete runs.csv header")
        with path.open("r+b") as handle:
            handle.truncate(end)
        print("Discarded one unfinished CSV append; its trajectory will restart.", flush=True)
    records = {}
    for row in csv_rows(path):
        if set(row) != set(RAW_FIELDS) or any(v is None for v in row.values()):
            raise ValueError("Malformed raw CSV record")
        for key in FLOAT_FIELDS:
            row[key] = float(row[key]) if row[key] else None
        for key in INT_FIELDS:
            row[key] = int(row[key])
        validate_record(row)
        key = record_key(row)
        if key in records:
            raise ValueError(f"Duplicate completed trajectory: {key}")
        records[key] = row
    return records


def append_record(handle, writer, records, row):
    key = record_key(row)
    if key in records:
        raise ValueError("Refusing to append a duplicate trajectory")
    validate_record(row)
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())
    records[key] = row


def protocol():
    return dict(
        schema="section-7.2.2-paper-v1", N=N_SAMPLES, sampling=SAMPLING,
        initial_region=[[-.8, -.5], [1.2, 1.5]],
        methods=list(ETA0), etas=list(ETAS), tau=TAU, J=J, budget=NMAX,
        safeguard_norm=GUARD, preconditioner=dict(alpha=CFG.alpha, a1=CFG.a1,
                                                   beta=1-CFG.a1, epsilon=CFG.eps_M),
        certification="raw gradient norm < 1e-6; lambda1 < -1e-6; lambda2 > 1e-6",
        endpoints=ENDPOINTS, endpoint_radius=1e-5,
        success_policy="Every certified index-1 endpoint counts, including an unexpected S_other.",
        scan_policy="All 41 steps, both methods, all 512 starts; no outcome-dependent skipping.",
        selection_context="The manuscript grid and SI parameters were selected after prior exploration; this script reproduces the fixed manuscript experiment.",
        engine="original_numpy_batch_float64", blas_threads=1,
    )


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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_publication_data(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def initialize_output(output):
    expected = _publication_data(json.loads(json.dumps(protocol())))
    manifest = output / "protocol.json"
    if manifest.exists():
        actual = json.loads(manifest.read_text(encoding="utf-8"))
        if _publication_data(actual.get("experiment")) != expected:
            raise ValueError("Existing data use different experiment parameters; choose a new --output.")
        environment = actual.get("environment", {})
        current = runtime_environment()
        if any(environment.get(k) != current[k] for k in current):
            raise ValueError("The saved numerical environment differs; choose a new --output to avoid mixing environments.")
    else:
        checkpoints = output / "_checkpoints"
        if checkpoints.exists() and any(checkpoints.iterdir()):
            raise ValueError("Existing checkpoints have no matching protocol; choose a new --output.")
        write_json(manifest, dict(experiment=expected, environment=runtime_environment(),
                                 created_utc=datetime.now(timezone.utc).isoformat()))
    (output / "_checkpoints").mkdir(exist_ok=True)
    write_csv(output / "initial_points.csv", ["start_id", "x0", "y0"],
              [dict(start_id=i, x0=x, y0=y) for i, (x, y) in enumerate(initial_points())])


def runtime_environment():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, sympy=sympy.__version__, platform=platform.platform(),
                machine=platform.machine())


def checkpoint_path(output, method, eta):
    return output / "_checkpoints" / f"{method}_eta_{ETAS.index(eta):02d}.csv"


# Completed trajectories are scientific records; resume only missing starts.
def load_checkpoints(output):
    records = {}
    expected = {checkpoint_path(output, method, eta) for method, eta in configs()}
    unknown = set((output / "_checkpoints").glob("*.csv")) - expected
    if unknown:
        raise ValueError(f"Unrecognized checkpoint files: {sorted(unknown)}")
    for method, eta in configs():
        rows = read_records(checkpoint_path(output, method, eta))
        if any((row["method"], row["eta"]) != (method, eta) for row in rows.values()):
            raise ValueError("Checkpoint contains a different configuration.")
        records.update(rows)
    return records


# Summarize complete groups of 512 starts, including failures and all certified saddles.
def save_tables(output, records):
    table, progress = [], []
    for method, eta in configs():
        rows = [records[(method, eta, i)] for i in range(N_SAMPLES)
                if (method, eta, i) in records]
        progress.append(dict(method=method, eta=eta, completed=len(rows), planned=N_SAMPLES,
                             status="COMPLETE" if len(rows) == N_SAMPLES else "PENDING"))
        if len(rows) != N_SAMPLES:
            continue
        endpoint_counts = Counter(row["endpoint"] for row in rows)
        categories = Counter(row["terminal_category"] for row in rows)
        success = categories["success_index1"]
        if success != sum(endpoint_counts[label] for label in (*ENDPOINTS, "S_other")):
            raise ValueError("Endpoint totals disagree with certification.")
        if success + endpoint_counts["U"] != N_SAMPLES:
            raise ValueError("Success and failure totals do not sum to 512.")
        table.append(dict(method=method, eta=eta, tau=TAU, N=N_SAMPLES, sampling=SAMPLING,
                          budget=NMAX, N_H=endpoint_counts["S_H"], N_L=endpoint_counts["S_L"],
                          N_other=endpoint_counts["S_other"], success=success,
                          unsuccessful=endpoint_counts["U"], success_percent=100*success/N_SAMPLES,
                          high_hit_percent=100*endpoint_counts["S_H"]/N_SAMPLES,
                          low_hit_percent=100*endpoint_counts["S_L"]/N_SAMPLES,
                          **{key: categories[key] for key in CATEGORIES[1:]}))
    write_csv(output / "table_main.csv", MAIN_FIELDS, table)
    write_csv(output / "progress.csv", ["method", "eta", "completed", "planned", "status"], progress)
    return table


class CancellationFlag:
    def __init__(self, event):
        self.event = event

    def __bool__(self):
        return self.event.is_set()


def initialize_worker(event):
    global CANCEL
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    CANCEL = CancellationFlag(event)


def run_configuration(output, method, eta):
    if CANCEL:
        return []
    path = checkpoint_path(output, method, eta)
    if not path.exists():
        write_csv(path, RAW_FIELDS, [])
    records = read_records(path)
    pending = [i for i in range(N_SAMPLES) if (method, eta, i) not in records]
    if not pending:
        return list(records.values())
    starts = initial_points()
    print(f"START {method}, eta={eta:.8g}: {len(pending)} starts pending", flush=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)

        def completed(local_id, result):
            i = pending[local_id]
            row = dict(method=method, eta=eta, tau=TAU, budget=NMAX, start_id=i,
                       x0=float(starts[i, 0]), y0=float(starts[i, 1]), **result)
            append_record(handle, writer, records, prepare_record(row, "run.py:numpy-batch"))

        def progress(step, active):
            if step % 50000 == 0:
                print(f"PROGRESS {method}, eta={eta:.8g}: updates={step}, active={active}", flush=True)

        run_batch(sys.modules[__name__], starts[pending], method, eta, TAU, NMAX,
                  on_complete=completed, on_progress=progress)
    if not CANCEL and len(records) != N_SAMPLES:
        raise RuntimeError("Incomplete configuration after batch termination.")
    if len(records) == N_SAMPLES:
        labels = Counter(row["endpoint"] for row in records.values())
        print(f"DONE {method}, eta={eta:.8g}: S_H={labels['S_H']}, S_L={labels['S_L']}, "
              f"other={labels['S_other']}, unsuccessful={labels['U']}", flush=True)
    return list(records.values())


def compute_missing(output, records, workers):
    pending = [(method, eta) for method, eta in configs()
               if any((method, eta, i) not in records for i in range(N_SAMPLES))]
    if not pending:
        return
    try:
        if workers == 1:
            for method, eta in pending:
                records.update((record_key(row), row) for row in run_configuration(output, method, eta))
                save_tables(output, records)
        else:
            context = multiprocessing.get_context("spawn")
            event = context.Event()
            previous_handler = signal.getsignal(signal.SIGINT)
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                         initializer=initialize_worker, initargs=(event,)) as pool:
                    futures = []
                    try:
                        futures = [pool.submit(run_configuration, output, method, eta)
                                   for method, eta in pending]
                        for future in as_completed(futures):
                            records.update((record_key(row), row) for row in future.result())
                            save_tables(output, records)
                    except BaseException:
                        signal.signal(signal.SIGINT, signal.SIG_IGN)
                        event.set()
                        for future in futures:
                            future.cancel()
                        raise
            finally:
                signal.signal(signal.SIGINT, previous_handler)
    finally:
        records.clear()
        records.update(load_checkpoints(output))
        save_tables(output, records)


def paper_outputs(output, table):
    if len(table) != 2*len(ETAS):
        raise ValueError("The full 82-configuration scan is required for paper outputs.")
    summary = {}
    for method in ETA0:
        rows = [row for row in table if row["method"] == method]
        example = next(row for row in rows if math.isclose(row["eta"], 1e-4, rel_tol=1e-12))
        summary[method] = dict(
            overall_success_percent_range=[min(row["success_percent"] for row in rows),
                                           max(row["success_percent"] for row in rows)],
            low_saddle_percent_range=[min(row["low_hit_percent"] for row in rows),
                                      max(row["low_hit_percent"] for row in rows)],
            eta_1e_4={key: example[key] for key in ("eta", "N_H", "N_L", "N_other", "unsuccessful")},
        )
    summary["unexpected_certified_saddles"] = sum(row["N_other"] for row in table)
    summary["interpretation"] = ("All index-1 certified saddles count toward overall success. "
                                 "If N_other is nonzero, update the manuscript's two-saddle interpretation.")
    write_json(output / "paper_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    styles = {"HiSD": dict(color="#D55E00", marker="o"),
              "SI-pHiSD": dict(color="#0072B2", marker="v")}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 17,
                         "legend.fontsize": 15, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for method in ETA0:
            rows = sorted((row for row in table if row["method"] == method), key=lambda row: row["eta"])
            ax.plot([row["eta"] for row in rows], [row["success_percent"] for row in rows],
                    label=method, linewidth=2, markersize=6, markerfacecolor="none",
                    clip_on=False, **styles[method])
        ax.set_xscale("log")
        ax.set_xlabel(r"Step size $\eta$")
        ax.set_ylabel("Overall success rate (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(ETAS[0], ETAS[-1]*10**.12)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", alpha=.3)
        ax.set_axisbelow(True)
        ax.legend(loc="center right", frameon=True, framealpha=.8)
        ax.set_title("Modified Müller Potential")
        fig.tight_layout()
        fig.savefig(output / "2.2.pdf", bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def smoke_test():
    solver = sys.modules[__name__]
    points = initial_points()[[0, 255, 511]]
    for method in ETA0:
        results = run_batch(solver, points, method, 1e-4, TAU, budget=5)
        if len(results) != len(points):
            raise RuntimeError("Smoke check returned the wrong number of starts.")
        for result in results:
            if result["execution_status"] != "completed" or result["n_outer_updates"] > 5:
                raise RuntimeError("Smoke check did not respect its update limit.")
            if result["terminal_category"] not in CATEGORIES:
                raise RuntimeError("Smoke check returned an invalid certification.")
    for point in ENDPOINTS.values():
        if certify(point)["terminal_category"] != "success_index1":
            raise RuntimeError("Reference saddle does not meet endpoint certification.")
    print("Smoke check passed: both methods, six bounded trajectories, both reference saddles. No paper outputs written.")


def _lock_output(lock):
    if os.name == "nt":
        import msvcrt
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise BlockingIOError("Output directory is already locked") from exc
            raise
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory; relative paths are relative to the current working directory")
    parser.add_argument("--workers", type=int, default=3, help="Parallel configurations (default: 3)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print the fixed protocol; no files or simulations")
    modes.add_argument("--smoke-test", action="store_true", help="Check six short trajectories; no paper outputs")
    modes.add_argument("--plot-only", action="store_true", help="Regenerate paper outputs from a complete saved scan")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.dry_run:
        print(json.dumps(_publication_data(protocol()), ensure_ascii=False, indent=2))
        return
    if args.smoke_test:
        smoke_test()
        return
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phisd-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    output = args.output.expanduser().resolve()
    if args.plot_only and not (output / "protocol.json").is_file():
        parser.error("--plot-only requires an existing complete output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".scan.lock").open("a+b") as lock:
        try:
            _lock_output(lock)
        except BlockingIOError:
            parser.error("Another process is already using this output directory")
        initialize_output(output)
        records = load_checkpoints(output)
        print(f"Section 7.2.2: {2*len(ETAS)*N_SAMPLES} trajectories; "
              f"{len(records)} complete; {NMAX} updates maximum per start. Output: {output}", flush=True)
        if not args.plot_only:
            compute_missing(output, records, args.workers)
        table = save_tables(output, records)
        paper_outputs(output, table)
        print(f"Complete: {output / '2.2.pdf'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Completed starts are saved; rerun the same command to resume.", flush=True)
        raise SystemExit(130)
