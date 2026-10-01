from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


@dataclass
class HPolytope:
    """Bounded convex polytope P={x | Hx<=h}, with utilities focused on 2-D examples."""
    H: np.ndarray
    h: np.ndarray

    def __post_init__(self):
        self.H = np.asarray(self.H, float)
        self.h = np.asarray(self.h, float).reshape(-1)
        if self.H.ndim != 2 or self.H.shape[0] != self.h.size:
            raise ValueError("Inconsistent H/h dimensions")

    @classmethod
    def box(cls, lower, upper):
        lower = np.asarray(lower, float); upper = np.asarray(upper, float)
        n = lower.size
        H = np.vstack([np.eye(n), -np.eye(n)])
        h = np.r_[upper, -lower]
        return cls(H, h)

    @classmethod
    def from_vertices_2d(cls, V):
        V = np.asarray(V, float)
        hull = ConvexHull(V)
        # scipy hull.equations: normal*x + offset <= 0
        H = hull.equations[:, :2]
        h = -hull.equations[:, 2]
        return cls(H, h).remove_redundancy()

    def contains(self, x, tol=1e-8):
        x = np.asarray(x, float)
        return bool(np.all(self.H @ x <= self.h + tol))

    def intersect(self, other: "HPolytope"):
        return HPolytope(np.vstack([self.H, other.H]), np.r_[self.h, other.h]).remove_redundancy()

    def vertices_2d(self, tol=1e-8):
        if self.H.shape[1] != 2:
            raise ValueError("vertices_2d only supports 2D")
        pts = []
        for i, j in itertools.combinations(range(len(self.h)), 2):
            M = self.H[[i, j]]
            if abs(np.linalg.det(M)) < 1e-10:
                continue
            x = np.linalg.solve(M, self.h[[i, j]])
            if np.all(self.H @ x <= self.h + tol):
                pts.append(x)
        if len(pts) < 3:
            return np.asarray(pts)
        pts = np.unique(np.round(np.asarray(pts), 10), axis=0)
        if len(pts) < 3:
            return pts
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    def remove_redundancy(self, tol=1e-8):
        H, h = self.H, self.h
        keep = []
        n = H.shape[1]
        for i in range(H.shape[0]):
            idx = [j for j in range(H.shape[0]) if j != i]
            if not idx:
                keep.append(i); continue
            # Is row i implied by all others? Maximize H_i x under other rows.
            res = linprog(-H[i], A_ub=H[idx], b_ub=h[idx], bounds=[(None, None)]*n,
                          method="highs")
            if (not res.success) or (-res.fun > h[i] + tol):
                keep.append(i)
        return HPolytope(H[keep], h[keep])

    def approximately_equal(self, other: "HPolytope", tol=1e-6):
        if self.H.shape[1] != 2 or other.H.shape[1] != 2:
            return False
        V1, V2 = self.vertices_2d(), other.vertices_2d()
        if len(V1) == 0 or len(V2) == 0: return len(V1) == len(V2)
        return all(other.contains(v, tol) for v in V1) and all(self.contains(v, tol) for v in V2)


def linear_preimage(S: HPolytope, A: np.ndarray):
    """Pre(S) for autonomous x+=Ax."""
    return HPolytope(S.H @ np.asarray(A, float), S.h.copy()).remove_redundancy()


def predecessor_scalar_input(S: HPolytope, A, B, umin, umax):
    """
    Exact H-representation of Pre(S) for x+=Ax+Bu and scalar u in [umin,umax]
    via Fourier-Motzkin elimination of u.
    """
    A = np.asarray(A, float); B = np.asarray(B, float).reshape(-1, 1)
    ax = S.H @ A
    bu = (S.H @ B).reshape(-1)
    c = S.h.copy()
    # input bounds as inequalities: u<=umax, -u<=-umin
    ax = np.vstack([ax, np.zeros((2, A.shape[1]))])
    bu = np.r_[bu, 1.0, -1.0]
    c = np.r_[c, umax, -umin]

    pos = np.where(bu > 1e-12)[0]
    neg = np.where(bu < -1e-12)[0]
    zer = np.where(np.abs(bu) <= 1e-12)[0]
    Hnew, hnew = [], []
    for i in zer:
        Hnew.append(ax[i]); hnew.append(c[i])
    for nidx in neg:
        for pidx in pos:
            # lower(nidx) <= upper(pidx)
            row = ax[pidx] / bu[pidx] - ax[nidx] / bu[nidx]
            rhs = c[pidx] / bu[pidx] - c[nidx] / bu[nidx]
            Hnew.append(row); hnew.append(rhs)
    return HPolytope(np.asarray(Hnew), np.asarray(hnew)).remove_redundancy()


def maximal_invariant(A, X: HPolytope, max_iter=100, tol=1e-7):
    """Maximal positively invariant subset of X for autonomous x+=Ax."""
    Omega = X.remove_redundancy()
    history = [Omega]
    for _ in range(max_iter):
        nxt = X.intersect(linear_preimage(Omega, A))
        history.append(nxt)
        if nxt.approximately_equal(Omega, tol):
            return nxt, history
        Omega = nxt
    return Omega, history


def maximal_control_invariant(A, B, X: HPolytope, umin, umax, max_iter=100, tol=1e-7):
    """Maximal control-invariant subset of X for scalar constrained input."""
    C = X.remove_redundancy()
    history = [C]
    for _ in range(max_iter):
        pre = predecessor_scalar_input(C, A, B, umin, umax)
        nxt = X.intersect(pre)
        history.append(nxt)
        if nxt.approximately_equal(C, tol):
            return nxt, history
        C = nxt
    return C, history

def affine_image_2d(P: HPolytope, M: np.ndarray, c=None):
    """Affine image {M x + c | x in P} for a full-dimensional 2-D result."""
    M = np.asarray(M, float)
    if c is None:
        c = np.zeros(M.shape[0])
    c = np.asarray(c, float).reshape(-1)
    V = P.vertices_2d()
    Y = (M @ V.T).T + c
    return HPolytope.from_vertices_2d(Y)


def minkowski_sum_2d(P: HPolytope, Q: HPolytope):
    """Minkowski sum P (+) Q using vertex enumeration in 2-D."""
    VP = P.vertices_2d()
    VQ = Q.vertices_2d()
    Vsum = np.asarray([p + q for p in VP for q in VQ])
    return HPolytope.from_vertices_2d(Vsum)


def reachable_set_scalar_input_2d(S: HPolytope, A, B, umin, umax):
    """
    One-step forward reachable set
        Post(S) = {A x + B u | x in S, u in [umin, umax]}
    for a 2-D state and scalar input.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float).reshape(-1)
    V = S.vertices_2d()
    pts = []
    for x in V:
        for u in (umin, umax):
            pts.append(A @ x + B * u)
    return HPolytope.from_vertices_2d(np.asarray(pts))


def closed_loop_admissible_set_scalar_input(X: HPolytope, K, umin, umax):
    """
    States in X for which u=-Kx respects scalar input bounds.
    """
    K = np.asarray(K, float).reshape(1, -1)
    Hu = np.vstack([-K, K])
    hu = np.array([umax, -umin], float)
    return X.intersect(HPolytope(Hu, hu))


def n_step_controllable_sets(A, B, X: HPolytope, target: HPolytope,
                             umin, umax, N):
    """
    K_0 = target,
    K_{i+1} = X intersect Pre_U(K_i).
    """
    sets = [target.intersect(X)]
    for _ in range(N):
        pre = predecessor_scalar_input(sets[-1], A, B, umin, umax)
        sets.append(X.intersect(pre))
    return sets

