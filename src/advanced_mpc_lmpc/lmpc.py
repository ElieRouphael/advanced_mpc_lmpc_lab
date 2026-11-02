from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

from .linear import prediction_matrices
from .qp import solve_qp


def state_penalty(x, Q, Q1=None):
    """x'Qx + ||Q1 x||_1."""
    x = np.asarray(x, float).reshape(-1)
    Q = np.asarray(Q, float)
    value = float(x @ Q @ x)

    if Q1 is not None:
        Q1 = np.asarray(Q1, float)
        if Q1.size:
            value += float(np.linalg.norm(Q1 @ x, 1))

    return value


def stage_cost(x, u, Q, R, Q1=None):
    """h(x,u)=x'Qx+u'Ru+||Q1 x||_1."""
    x = np.asarray(x, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    R = np.atleast_2d(np.asarray(R, float))

    return state_penalty(x, Q, Q1) + float(u @ R @ u)


def trajectory_cost_to_go(X, U, Q, R, Q1=None):
    """
    Roll-out cost-to-go for a successful finite trajectory.

    X = [x_0,...,x_T], U = [u_0,...,u_{T-1}]
    J_t = sum_{k=t}^{T-1} h(x_k,u_k), J_T=0.
    """
    X = np.asarray(X, float)
    U = np.asarray(U, float)

    if len(X) != len(U) + 1:
        raise ValueError("X must contain one more state sample than U")

    J = np.zeros(len(X))

    for k in range(len(U) - 1, -1, -1):
        J[k] = stage_cost(X[k], U[k], Q, R, Q1) + J[k + 1]

    return J


@dataclass
class FTOCPSolution:
    success: bool
    U: np.ndarray
    X: np.ndarray
    objective: float
    message: str
    max_violation: float
    raw_decision: np.ndarray


def solve_ftocp_l1(
    A,
    B,
    Q,
    R,
    Q1,
    N,
    x0,
    x_lower,
    x_upper,
    u_lower,
    u_upper,
    warm=None,
):
    """
    Convex finite-horizon problem of the repeated double-integrator task (notebook 05):

        min sum_{k=0}^{N-1}
            x_k' Q x_k + u_k' R u_k + ||Q1 x_k||_1

        s.t.
            x_{k+1}=A x_k+B u_k
            x_k in X
            u_k in U.

    The state trajectory is condensed out. The L1 term is represented with
    epigraph variables, so the result remains a convex QP with linear
    constraints.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Q = np.asarray(Q, float)
    R = np.atleast_2d(np.asarray(R, float))
    Q1 = np.asarray(Q1, float)

    x0 = np.asarray(x0, float).reshape(-1)
    x_lower = np.asarray(x_lower, float).reshape(-1)
    x_upper = np.asarray(x_upper, float).reshape(-1)
    u_lower = np.asarray(u_lower, float).reshape(-1)
    u_upper = np.asarray(u_upper, float).reshape(-1)

    N = int(N)
    n = A.shape[0]
    m = B.shape[1]
    p = Q1.shape[0]

    Sx, Su = prediction_matrices(A, B, N)

    nu = N * m
    ne = max(N - 1, 0) * p
    nz = nu + ne

    Qbar = np.kron(np.eye(N), Q)
    Qbar[-n:, -n:] = 0.0
    Rbar = np.kron(np.eye(N), R)

    H = np.zeros((nz, nz))
    f = np.zeros(nz)

    H[:nu, :nu] = 2.0 * (Su.T @ Qbar @ Su + Rbar)
    f[:nu] = 2.0 * Su.T @ Qbar @ (Sx @ x0)

    if ne:
        f[nu:] = 1.0

    A_ub = []
    b_ub = []

    # Predicted state bounds x_1,...,x_N.
    A_state = np.zeros((2 * N * n, nz))
    A_state[:N*n, :nu] = Su
    A_state[N*n:, :nu] = -Su

    b_state = np.r_[
        np.tile(x_upper, N) - Sx @ x0,
        -np.tile(x_lower, N) + Sx @ x0,
    ]

    A_ub.append(A_state)
    b_ub.append(b_state)

    # L1 epigraph for x_1,...,x_{N-1}; x_0 is constant.
    for k in range(1, N):
        row_x = Su[(k-1)*n:k*n, :]
        offset_x = Sx[(k-1)*n:k*n, :] @ x0

        for i in range(p):
            e_idx = nu + (k - 1) * p + i

            row = np.zeros(nz)
            row[:nu] = Q1[i] @ row_x
            row[e_idx] = -1.0
            A_ub.append(row[None, :])
            b_ub.append(np.array([-Q1[i] @ offset_x]))

            row = np.zeros(nz)
            row[:nu] = -Q1[i] @ row_x
            row[e_idx] = -1.0
            A_ub.append(row[None, :])
            b_ub.append(np.array([Q1[i] @ offset_x]))

    lb = np.r_[
        np.tile(u_lower, N),
        np.zeros(ne),
    ]

    ub = np.r_[
        np.tile(u_upper, N),
        np.full(ne, np.inf),
    ]

    if warm is None or len(np.asarray(warm).reshape(-1)) != nz:
        warm = np.r_[np.zeros(nu), np.zeros(ne)]

    res = solve_qp(
        H,
        f,
        A_ub=np.vstack(A_ub),
        b_ub=np.concatenate(b_ub),
        lb=lb,
        ub=ub,
        x0=np.asarray(warm, float).reshape(-1),
        tol=1e-9,
        maxiter=2000,
    )

    U = res.x[:nu].reshape(N, m)
    Xpred = (Sx @ x0 + Su @ res.x[:nu]).reshape(N, n)
    X = np.vstack([x0, Xpred])

    objective = sum(
        stage_cost(X[k], U[k], Q, R, Q1)
        for k in range(N)
    )

    return FTOCPSolution(
        success=bool(res.success),
        U=U,
        X=X,
        objective=float(objective),
        message=res.message,
        max_violation=float(res.max_violation),
        raw_decision=res.x.copy(),
    )


def convex_safe_value(x, safe_states, safe_values):
    """
    Convex-envelope terminal value

        V_c(x) = min lambda' V
        s.t. S' lambda = x,
             1' lambda = 1,
             lambda >= 0.

    Returns (value, lambda, success).
    """
    x = np.asarray(x, float).reshape(-1)
    S = np.asarray(safe_states, float)
    V = np.asarray(safe_values, float).reshape(-1)

    if len(S) != len(V):
        raise ValueError("safe_states and safe_values must have the same length")

    A_eq = np.vstack([S.T, np.ones((1, len(S)))])
    b_eq = np.r_[x, 1.0]

    res = linprog(
        c=V,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * len(S),
        method="highs",
    )

    if not res.success:
        return np.inf, np.zeros(len(S)), False

    return float(res.fun), np.asarray(res.x, float), True


@dataclass
class LMPCSolution:
    success: bool
    U: np.ndarray
    X: np.ndarray
    lambdas: np.ndarray
    objective: float
    terminal_state: np.ndarray
    message: str
    max_violation: float
    raw_decision: np.ndarray
    safe_states: np.ndarray
    safe_values: np.ndarray
    safe_inputs: np.ndarray
    next_index: np.ndarray


class LMPCController:
    """
    Educational convex-safe-set LMPC for deterministic LTI repetitive tasks.

    The controller stores successful trajectories. With the stored safe states
    as the rows of S, at each online solve

        x_N = S' lambda,
        lambda >= 0,
        1'lambda = 1,

    constrains the terminal state to the convex hull of stored safe states.
    The terminal cost is the corresponding convex combination of stored
    realized costs-to-go.

    A target equilibrium x=0 with u=0 is included explicitly.
    """

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        N,
        x_lower,
        x_upper,
        u_lower,
        u_upper,
        Q1=None,
        target=None,
        target_tolerance=1e-7,
    ):
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.Q = np.asarray(Q, float)
        self.R = np.atleast_2d(np.asarray(R, float))
        self.Q1 = (
            np.zeros((self.A.shape[0], self.A.shape[0]))
            if Q1 is None
            else np.asarray(Q1, float)
        )

        self.N = int(N)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]

        self.x_lower = np.asarray(x_lower, float).reshape(self.n)
        self.x_upper = np.asarray(x_upper, float).reshape(self.n)
        self.u_lower = np.asarray(u_lower, float).reshape(self.m)
        self.u_upper = np.asarray(u_upper, float).reshape(self.m)

        self.target = (
            np.zeros(self.n)
            if target is None
            else np.asarray(target, float).reshape(self.n)
        )
        self.target_tolerance = float(target_tolerance)

        self.Sx, self.Su = prediction_matrices(self.A, self.B, self.N)

        self.trajectories = []
        self.inputs = []
        self.qfun = []
        self.iteration_costs = []

    def add_trajectory(self, X, U):
        X = np.asarray(X, float)
        U = np.asarray(U, float)

        if len(X) != len(U) + 1:
            raise ValueError("X must contain one more state sample than U")

        if X.shape[1] != self.n or U.shape[1] != self.m:
            raise ValueError("trajectory dimensions do not match the controller")

        if np.linalg.norm(X[-1] - self.target) > self.target_tolerance:
            raise ValueError(
                "trajectory does not terminate sufficiently close to the target"
            )

        J = trajectory_cost_to_go(X, U, self.Q, self.R, self.Q1)

        self.trajectories.append(X.copy())
        self.inputs.append(U.copy())
        self.qfun.append(J.copy())
        self.iteration_costs.append(float(J[0]))

    def safe_data(self, last_n_trajectories=None):
        """
        Return safe states, sampled cost-to-go, continuation inputs, and
        successor indices.

        The target is stored once explicitly at index 0. Terminal copies of
        the target from each trajectory are omitted to reduce degeneracy.
        """
        if last_n_trajectories is None:
            indices = range(len(self.trajectories))
        else:
            q = max(int(last_n_trajectories), 1)
            start = max(0, len(self.trajectories) - q)
            indices = range(start, len(self.trajectories))

        S = [self.target.copy()]
        V = [0.0]
        Usafe = [np.zeros(self.m)]
        next_index = [0]

        for it in indices:
            X = self.trajectories[it]
            U = self.inputs[it]
            J = self.qfun[it]

            T = len(U)
            start_idx = len(S)

            for k in range(T):
                S.append(X[k].copy())
                V.append(float(J[k]))
                Usafe.append(U[k].copy())

                if k < T - 1:
                    next_index.append(start_idx + k + 1)
                else:
                    next_index.append(0)

        return (
            np.asarray(S, float),
            np.asarray(V, float),
            np.asarray(Usafe, float),
            np.asarray(next_index, int),
        )

    def _pack_warm(self, x0, U, lam, safe_states):
        x0 = np.asarray(x0, float).reshape(self.n)
        U = np.asarray(U, float).reshape(self.N, self.m)
        lam = np.asarray(lam, float).reshape(len(safe_states))

        nu = self.N * self.m
        p = self.Q1.shape[0]
        ne = max(self.N - 1, 0) * p

        if ne:
            X = (
                self.Sx @ x0
                + self.Su @ U.reshape(-1)
            ).reshape(self.N, self.n)

            e = []
            for k in range(self.N - 1):
                e.extend(np.abs(self.Q1 @ X[k]))

            return np.r_[U.reshape(-1), lam, np.asarray(e, float)]

        return np.r_[U.reshape(-1), lam]

    def warm_start_from_stored_path(
        self,
        x0,
        last_n_trajectories=None,
        match_tolerance=1e-6,
    ):
        S, V, Usafe, nxt = self.safe_data(last_n_trajectories)

        x0 = np.asarray(x0, float).reshape(self.n)
        dist = np.linalg.norm(S - x0, axis=1)
        idx = int(np.argmin(dist))

        if dist[idx] > match_tolerance:
            return None

        controls = []
        current = idx

        for _ in range(self.N):
            controls.append(Usafe[current].copy())
            current = int(nxt[current])

        lam = np.zeros(len(S))
        lam[current] = 1.0

        return self._pack_warm(
            x0,
            np.asarray(controls),
            lam,
            S,
        )

    def shifted_warm_start(self, x_next, solution):
        lam = solution.lambdas
        Usafe = solution.safe_inputs
        nxt = solution.next_index
        S = solution.safe_states

        u_terminal = lam @ Usafe

        U_shift = np.vstack([
            solution.U[1:],
            u_terminal.reshape(1, -1),
        ])

        lam_shift = np.zeros(len(S))
        np.add.at(lam_shift, nxt, lam)

        return self._pack_warm(
            np.asarray(x_next, float),
            U_shift,
            lam_shift,
            S,
        )

    def solve(
        self,
        x0,
        *,
        last_n_trajectories=None,
        warm=None,
    ):
        x0 = np.asarray(x0, float).reshape(self.n)

        S, V, Usafe, nxt = self.safe_data(last_n_trajectories)

        M = len(S)
        nu = self.N * self.m
        p = self.Q1.shape[0]
        ne = max(self.N - 1, 0) * p
        nz = nu + M + ne

        Qbar = np.kron(np.eye(self.N), self.Q)
        Qbar[-self.n:, -self.n:] = 0.0
        Rbar = np.kron(np.eye(self.N), self.R)

        H = np.zeros((nz, nz))
        f = np.zeros(nz)

        H[:nu, :nu] = 2.0 * (
            self.Su.T @ Qbar @ self.Su
            + Rbar
        )
        f[:nu] = 2.0 * self.Su.T @ Qbar @ (self.Sx @ x0)

        f[nu:nu+M] = V

        if ne:
            f[nu+M:] = 1.0

        A_ub = []
        b_ub = []

        A_state = np.zeros((2 * self.N * self.n, nz))
        A_state[:self.N*self.n, :nu] = self.Su
        A_state[self.N*self.n:, :nu] = -self.Su

        b_state = np.r_[
            np.tile(self.x_upper, self.N) - self.Sx @ x0,
            -np.tile(self.x_lower, self.N) + self.Sx @ x0,
        ]

        A_ub.append(A_state)
        b_ub.append(b_state)

        for k in range(1, self.N):
            row_x = self.Su[(k-1)*self.n:k*self.n, :]
            offset_x = self.Sx[(k-1)*self.n:k*self.n, :] @ x0

            for i in range(p):
                e_idx = nu + M + (k - 1) * p + i

                row = np.zeros(nz)
                row[:nu] = self.Q1[i] @ row_x
                row[e_idx] = -1.0
                A_ub.append(row[None, :])
                b_ub.append(np.array([-self.Q1[i] @ offset_x]))

                row = np.zeros(nz)
                row[:nu] = -self.Q1[i] @ row_x
                row[e_idx] = -1.0
                A_ub.append(row[None, :])
                b_ub.append(np.array([self.Q1[i] @ offset_x]))

        A_eq = np.zeros((self.n + 1, nz))
        b_eq = np.zeros(self.n + 1)

        A_eq[:self.n, :nu] = self.Su[-self.n:, :]
        A_eq[:self.n, nu:nu+M] = -S.T
        b_eq[:self.n] = -self.Sx[-self.n:, :] @ x0

        A_eq[self.n, nu:nu+M] = 1.0
        b_eq[self.n] = 1.0

        lb = np.r_[
            np.tile(self.u_lower, self.N),
            np.zeros(M),
            np.zeros(ne),
        ]

        ub = np.r_[
            np.tile(self.u_upper, self.N),
            np.ones(M),
            np.full(ne, np.inf),
        ]

        if warm is None:
            warm = self.warm_start_from_stored_path(
                x0,
                last_n_trajectories=last_n_trajectories,
            )

        if warm is None or len(np.asarray(warm).reshape(-1)) != nz:
            lam0 = np.zeros(M)
            nearest = int(np.argmin(np.linalg.norm(S - x0, axis=1)))
            lam0[nearest] = 1.0

            warm = self._pack_warm(
                x0,
                np.zeros((self.N, self.m)),
                lam0,
                S,
            )

        res = solve_qp(
            H,
            f,
            A_ub=np.vstack(A_ub),
            b_ub=np.concatenate(b_ub),
            A_eq=A_eq,
            b_eq=b_eq,
            lb=lb,
            ub=ub,
            x0=np.asarray(warm, float).reshape(-1),
            tol=1e-8,
            maxiter=3000,
        )

        U = res.x[:nu].reshape(self.N, self.m)
        lam = res.x[nu:nu+M]

        Xpred = (
            self.Sx @ x0
            + self.Su @ res.x[:nu]
        ).reshape(self.N, self.n)

        Xfull = np.vstack([x0, Xpred])

        objective = sum(
            stage_cost(
                Xfull[k],
                U[k],
                self.Q,
                self.R,
                self.Q1,
            )
            for k in range(self.N)
        )
        objective += float(lam @ V)

        terminal_state = S.T @ lam

        return LMPCSolution(
            success=bool(res.success),
            U=U,
            X=Xpred,
            lambdas=lam,
            objective=float(objective),
            terminal_state=terminal_state,
            message=res.message,
            max_violation=float(res.max_violation),
            raw_decision=res.x.copy(),
            safe_states=S,
            safe_values=V,
            safe_inputs=Usafe,
            next_index=nxt,
        )
