from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


@dataclass
class QPResult:
    x: np.ndarray
    success: bool
    objective: float
    message: str
    max_violation: float
    nit: int


def solve_qp(H, f=None, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
             lb=None, ub=None, x0=None, tol=1e-8, maxiter=1000):
    """
    Solve a small/medium convex QP
        min 0.5 z' H z + f' z
        s.t. A_ub z <= b_ub, A_eq z = b_eq, lb <= z <= ub.

    Uses SciPy/SLSQP so the educational repository has no external solver runtime.
    For large production MPC problems, use a dedicated QP solver (OSQP, qpOASES,
    HPIPM, GUROBI, ...).
    """
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
    if ub is None: ub =  np.inf * np.ones(n)
    bounds = Bounds(np.asarray(lb, float), np.asarray(ub, float))
    if x0 is None:
        x0 = np.zeros(n)
        finite_both = np.isfinite(bounds.lb) & np.isfinite(bounds.ub)
        x0[finite_both] = 0.5 * (bounds.lb[finite_both] + bounds.ub[finite_both])
        x0 = np.minimum(np.maximum(x0, np.where(np.isfinite(bounds.lb), bounds.lb, -1e6)),
                        np.where(np.isfinite(bounds.ub), bounds.ub, 1e6))

    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"ftol": tol, "maxiter": maxiter, "disp": False})

    violations = [0.0]
    if A_ub is not None: violations.append(float(np.max(A_ub @ res.x - b_ub)))
    if A_eq is not None: violations.append(float(np.max(np.abs(A_eq @ res.x - b_eq))))
    violations.append(float(np.max(np.maximum(bounds.lb - res.x, 0))))
    violations.append(float(np.max(np.maximum(res.x - bounds.ub, 0))))
    max_v = max(violations)
    success = bool(res.success and max_v <= 1e-5)
    return QPResult(res.x, success, float(res.fun), str(res.message), max_v, int(res.nit))
