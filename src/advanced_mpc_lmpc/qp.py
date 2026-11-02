"""A small, readable convex quadratic-program solver.

    minimise   0.5 z' H z + f' z
    subject to A_ub z <= b_ub,   A_eq z = b_eq,   lb <= z <= ub.

The solver is a primal-dual interior-point method on the KKT conditions of the
problem (chapter A3 of the course derives them, chapter A4 walks through this
file).  It is dense and written for problems with a few hundred variables, which
is every problem in this course; for large or embedded MPC use OSQP, qpOASES,
HPIPM or a commercial solver.  Everything is NumPy, so the course needs no
external solver runtime, and the same algorithm runs in the browser demos.

`solve_qp` returns the primal solution together with the Lagrange multipliers
of the inequalities (`lam`) and equalities (`nu`), the largest constraint
violation, the number of iterations and a `success` flag that means "the KKT
residuals are below the tolerance".  `solve_qp_slsqp` keeps the earlier
SciPy/SLSQP wrapper for comparison, and `feasibility_margin` certifies
feasibility with an independent linear program.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QPResult:
    x: np.ndarray
    success: bool
    objective: float
    message: str
    max_violation: float
    nit: int
    lam: np.ndarray = field(default_factory=lambda: np.zeros(0))   # multipliers of the stacked inequalities
    nu: np.ndarray = field(default_factory=lambda: np.zeros(0))    # multipliers of the equalities
    kkt_residual: float = np.nan                                   # largest KKT residual at the end


def stack_inequalities(n, A_ub=None, b_ub=None, lb=None, ub=None):
    """Return G, g with G z <= g: the user's rows, then the finite upper bounds, then the finite lower bounds."""
    rows, rhs = [], []
    if A_ub is not None:
        rows.append(np.asarray(A_ub, float).reshape(-1, n)); rhs.append(np.asarray(b_ub, float).reshape(-1))
    if ub is not None:
        ub = np.asarray(ub, float).reshape(-1)
        idx = np.where(np.isfinite(ub))[0]
        if idx.size:
            E = np.zeros((idx.size, n)); E[np.arange(idx.size), idx] = 1.0
            rows.append(E); rhs.append(ub[idx])
    if lb is not None:
        lb = np.asarray(lb, float).reshape(-1)
        idx = np.where(np.isfinite(lb))[0]
        if idx.size:
            E = np.zeros((idx.size, n)); E[np.arange(idx.size), idx] = -1.0
            rows.append(E); rhs.append(-lb[idx])
    if not rows:
        return np.zeros((0, n)), np.zeros(0)
    return np.vstack(rows), np.concatenate(rhs)


def solve_qp(H, f=None, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
             lb=None, ub=None, x0=None, tol=1e-9, maxiter=100, sigma=0.1, trace=None):
    """Solve the convex QP by a primal-dual interior-point method.

    H must be positive semidefinite (it is symmetrised).  `x0` is accepted for
    API compatibility and used as the starting primal point after projection
    into the bounds; an interior-point method does not need a warm start.

    The iteration solves the perturbed KKT system
        H z + f + G' lam + E' nu = 0,      G z + s = g,   s > 0,  lam > 0,
        E z = e,                           s_i lam_i = sigma * mu,
    by Newton's method, with the step shortened so that s and lam stay
    positive, and sigma = 0.1 pulling the complementarity products towards
    zero.  It stops when every residual is below `tol`, then polishes the
    answer: the constraints the iterate presses against (lam_i > s_i) are
    taken as the active set and the equality-constrained QP on that set is
    solved exactly; the polished point is kept if it is feasible and its
    multipliers are non-negative, which brings the solution to round-off.
    If `trace` is a list, every iterate (z, s, lam, residual, mu) is appended to it.
    """
    H = np.asarray(H, float)
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    f = np.zeros(n) if f is None else np.asarray(f, float).reshape(n)
    G, g = stack_inequalities(n, A_ub, b_ub, lb, ub)
    if A_eq is not None:
        E = np.asarray(A_eq, float).reshape(-1, n); e = np.asarray(b_eq, float).reshape(-1)
    else:
        E = np.zeros((0, n)); e = np.zeros(0)
    m, p = len(g), len(e)

    # starting point: z inside the bounds, slacks and multipliers strictly positive
    z = np.zeros(n) if x0 is None else np.asarray(x0, float).reshape(n).copy()
    if lb is not None:
        lo = np.asarray(lb, float).reshape(-1); z = np.where(np.isfinite(lo), np.maximum(z, lo), z)
    if ub is not None:
        hi = np.asarray(ub, float).reshape(-1); z = np.where(np.isfinite(hi), np.minimum(z, hi), z)
    lam = np.ones(m)
    s = np.maximum(g - G @ z, 1.0)
    nu = np.zeros(p)

    nit, resid, mu = 0, np.inf, 0.0
    for nit in range(1, maxiter + 1):
        rd = H @ z + f + G.T @ lam + E.T @ nu          # dual residual: stationarity of the Lagrangian
        rp = G @ z + s - g                             # primal residual of the inequalities
        re = E @ z - e                                 # primal residual of the equalities
        mu = (s @ lam) / m if m else 0.0               # average complementarity s_i * lam_i
        rc = sigma * mu - s * lam                      # centred complementarity residual
        resid = max(np.abs(rd).max(initial=0.0), np.abs(rp).max(initial=0.0), np.abs(re).max(initial=0.0))
        if trace is not None:
            trace.append((z.copy(), s.copy(), lam.copy(), resid, mu))
        if resid < tol and mu < tol:
            break
        # Newton step on the reduced system: s and lam are eliminated, leaving (dz, dnu)
        D = lam / s
        M = np.zeros((n + p, n + p))
        M[:n, :n] = H + G.T @ (D[:, None] * G)
        M[:n, n:] = E.T
        M[n:, :n] = E
        t = (rc + lam * rp) / s
        rhs = np.r_[-rd - G.T @ t, -re]
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M + 1e-12 * np.eye(n + p), rhs, rcond=None)[0]
        dz, dnu = sol[:n], sol[n:]
        ds = -rp - G @ dz
        dl = (rc - lam * ds) / s
        # fraction-to-the-boundary rule: keep s and lam strictly positive
        a = 1.0
        if m:
            neg_s, neg_l = ds < 0, dl < 0
            if neg_s.any(): a = min(a, float(np.min(-0.99 * s[neg_s] / ds[neg_s])))
            if neg_l.any(): a = min(a, float(np.min(-0.99 * lam[neg_l] / dl[neg_l])))
        zn, sn, ln, nn = z + a * dz, s + a * ds, lam + a * dl, nu + a * dnu
        if not (np.all(np.isfinite(zn)) and np.all(np.isfinite(sn)) and np.all(np.isfinite(ln)) and np.all(np.isfinite(nn))):
            break                                      # keep the last finite iterate for the polish
        z, s, lam, nu = zn, sn, ln, nn
        if a < 1e-12:
            break                                      # stalled at the boundary (a degenerate optimum): polish

    # polish: solve the equality-constrained QP on the active set, accept it if it is feasible and dual feasible.
    # It runs whenever the primal residuals are small, even if the dual ones are not: when more constraints are
    # active than there are variables (a degenerate optimum) the multipliers are not unique and the interior
    # point drifts among them, so the polish recovers one set by least squares, then non-negative least squares.
    prim = max((np.abs(G @ z + s - g).max() if m else 0.0), (np.abs(E @ z - e).max() if p else 0.0))
    if np.all(np.isfinite(z)) and prim < 1e-6:
        act = np.where((lam > s) | (s <= 1e-7 * (1.0 + np.abs(g))))[0] if m else np.zeros(0, int)   # pressed against, or touched
        Ga = G[act]
        k = len(act)
        Mp = np.zeros((n + k + p, n + k + p))
        Mp[:n, :n] = H; Mp[:n, n:n + k] = Ga.T; Mp[n:n + k, :n] = Ga
        Mp[:n, n + k:] = E.T; Mp[n + k:, :n] = E
        sol = np.linalg.lstsq(Mp, np.r_[-f, g[act], e], rcond=None)[0]
        zp, lp, nup = sol[:n], sol[n:n + k], sol[n + k:]
        if np.all(np.isfinite(zp)) and (not m or np.max(G @ zp - g) <= 1e-9) and (not p or np.max(np.abs(E @ zp - e)) <= 1e-9):
            grad = H @ zp + f
            if np.any(lp < -1e-9) or np.abs(grad + Ga.T @ lp + E.T @ nup).max(initial=0.0) > 1e-8:
                from scipy.optimize import nnls
                C = np.hstack([Ga.T, E.T, -E.T]) if (k or p) else np.zeros((n, 0))
                coef = nnls(C, -grad)[0] if C.shape[1] else np.zeros(0)
                lp, nup = coef[:k], coef[k:k + p] - coef[k + p:]
            rd = grad + Ga.T @ lp + E.T @ nup
            if np.all(lp >= -1e-9) and np.abs(rd).max(initial=0.0) <= 1e-7:
                z, nu = zp, nup
                lam = np.zeros(m); lam[act] = np.maximum(lp, 0.0)
                s = g - G @ z
                resid = max(np.abs(rd).max(initial=0.0), (np.abs(E @ z - e).max() if p else 0.0))
                mu = 0.0

    viol = [0.0]
    if m: viol.append(float(np.max(G @ z - g)))
    if p: viol.append(float(np.max(np.abs(E @ z - e))))
    max_v = max(viol)
    converged = bool(resid < 10 * tol and mu < 10 * tol and max_v <= 1e-7)
    message = "KKT conditions satisfied" if converged else f"not converged: residual {resid:.1e}, violation {max_v:.1e}"
    return QPResult(z, converged, float(0.5 * z @ H @ z + f @ z), message, max_v, nit, lam, nu, float(resid))


def solve_qp_slsqp(H, f=None, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                   lb=None, ub=None, x0=None, tol=1e-8, maxiter=1000):
    """The earlier SciPy/SLSQP wrapper, kept so that the two solvers can be compared."""
    from scipy.optimize import Bounds, LinearConstraint, minimize

    H = np.asarray(H, float)
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    f = np.zeros(n) if f is None else np.asarray(f, float).reshape(n)

    def obj(z): return 0.5 * z @ H @ z + f @ z
    def jac(z): return H @ z + f

    constraints = []
    if A_ub is not None:
        A_ub = np.asarray(A_ub, float); b_ub = np.asarray(b_ub, float).reshape(-1)
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
    if A_eq is not None:
        A_eq = np.asarray(A_eq, float); b_eq = np.asarray(b_eq, float).reshape(-1)
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    if lb is None: lb = -np.inf * np.ones(n)
    if ub is None: ub = np.inf * np.ones(n)
    bounds = Bounds(np.asarray(lb, float), np.asarray(ub, float))
    if x0 is None:
        x0 = np.zeros(n)
        finite_both = np.isfinite(bounds.lb) & np.isfinite(bounds.ub)
        x0[finite_both] = 0.5 * (bounds.lb[finite_both] + bounds.ub[finite_both])
        x0 = np.minimum(np.maximum(x0, np.where(np.isfinite(bounds.lb), bounds.lb, -1e6)),
                        np.where(np.isfinite(bounds.ub), bounds.ub, 1e6))
    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": tol, "maxiter": maxiter, "disp": False})
    violations = [0.0]
    if A_ub is not None: violations.append(float(np.max(A_ub @ res.x - b_ub)))
    if A_eq is not None: violations.append(float(np.max(np.abs(A_eq @ res.x - b_eq))))
    violations.append(float(np.max(np.maximum(bounds.lb - res.x, 0))))
    violations.append(float(np.max(np.maximum(res.x - bounds.ub, 0))))
    max_v = max(violations)
    success = bool(res.success and max_v <= 1e-5)
    return QPResult(res.x, success, float(res.fun), str(res.message), max_v, int(res.nit))


def feasibility_margin(n, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None):
    """Certify feasibility of a constraint set with an independent linear program.

    Returns the smallest s >= 0 such that some z satisfies A_ub z <= b_ub + s,
    |A_eq z - b_eq| <= s and lb - s <= z <= ub + s.  Zero (to tolerance)
    certifies that the constraints are feasible; a positive value certifies
    that they are not.  Uses SciPy's HiGHS, a solver that shares no code with
    `solve_qp`, so a QP failure can be told apart from an infeasible problem.
    """
    from scipy.optimize import linprog

    G, g = stack_inequalities(n, A_ub, b_ub, lb, ub)
    rows, rhs = [], []
    if len(g):
        rows.append(np.c_[G, -np.ones(len(g))]); rhs.append(g)
    if A_eq is not None:
        E = np.asarray(A_eq, float).reshape(-1, n); e = np.asarray(b_eq, float).reshape(-1)
        rows += [np.c_[E, -np.ones(len(e))], np.c_[-E, -np.ones(len(e))]]
        rhs += [e, -e]
    c = np.r_[np.zeros(n), 1.0]
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                  bounds=[(None, None)] * n + [(0, None)], method="highs")
    return float(res.x[-1]) if res.success else np.inf
