from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag

from .linear import prediction_matrices
from .qp import solve_qp


@dataclass
class CollisionSolution:
    strategy: str
    success: bool
    objective: float
    X: np.ndarray
    U: np.ndarray
    message: str
    max_violation: float
    conflict_steps: tuple
    d_yield: float
    d_pass: float
    terminal_stop: bool
    discretization: str


def conflict_bounds(d_bike=100.0, l_car=4.5, w_bike=8.0):
    """
    Collision-free longitudinal center-position bounds.

    The vehicle occupies [d-l_car/2, d+l_car/2].  The bicycle conflict
    region occupies [d_bike-w_bike/2, d_bike+w_bike/2].

    No overlap is guaranteed if either

        d <= d_bike - (l_car+w_bike)/2    (yield side)
    or
        d >= d_bike + (l_car+w_bike)/2    (pass side).
    """
    half_gap = 0.5 * (float(l_car) + float(w_bike))
    return float(d_bike - half_gap), float(d_bike + half_gap)


def vehicle_matrices(Ts=1.0, discretization="euler"):
    """
    State x=[d,v], input a.

    'euler' reproduces the supplied workshop:
        d+ = d + Ts v
        v+ = v + Ts a

    'zoh' is the exact zero-order-hold discretization of the continuous
    double integrator:
        d+ = d + Ts v + 0.5 Ts^2 a
        v+ = v + Ts a
    """
    Ts = float(Ts)
    A = np.array([[1.0, Ts],
                  [0.0, 1.0]])

    if discretization == "euler":
        B = np.array([[0.0],
                      [Ts]])
    elif discretization in {"zoh", "exact"}:
        B = np.array([[0.5 * Ts**2],
                      [Ts]])
    else:
        raise ValueError("discretization must be 'euler' or 'zoh'")

    return A, B


def _full_cost(X, U, *, v_ref, Qv, Ru, Pv):
    v = np.asarray(X, float)[:, 1]
    a = np.asarray(U, float).reshape(-1)
    N = len(a)

    return float(
        Ru * np.sum(a**2)
        + Qv * np.sum((v[:N] - v_ref)**2)
        + Pv * (v[N] - v_ref)**2
    )


def _empty_solution(strategy, x0, *, message, conflict_steps, d_yield,
                    d_pass, terminal_stop, discretization):
    return CollisionSolution(
        strategy=strategy,
        success=False,
        objective=np.inf,
        X=np.asarray(x0, float).reshape(1, 2),
        U=np.empty((0, 1)),
        message=message,
        max_violation=np.inf,
        conflict_steps=tuple(conflict_steps),
        d_yield=float(d_yield),
        d_pass=float(d_pass),
        terminal_stop=bool(terminal_stop),
        discretization=discretization,
    )


def solve_collision_strategy(
    x0,
    strategy,
    *,
    N=4,
    Ts=1.0,
    d_bike=100.0,
    l_car=4.5,
    w_bike=8.0,
    conflict_steps=None,
    mu=0.10,
    g=9.81,
    v_max=32.0,
    v_ref=20.0,
    Qv=3.0,
    Ru=1.0,
    Pv=3.0,
    yield_boundary="geometric",
    terminal_stop=False,
    discretization="euler",
    warm=None,
):
    """
    Solve one continuous convex QP corresponding to one logical strategy.

    strategy:
        'yield' -> stay on the near side of the conflict region
        'pass'  -> clear the far side of the conflict region

    The default N=4, Ts=1, Qv=3, Ru=1, Pv=3, v_ref=20, mu=0.10,
    v_max=32 and Euler dynamics reproduce the numerical setup of the
    supplied workshop, except that the default yield boundary uses the
    geometrically collision-free d_max from the accompanying slides.

    Parameters
    ----------
    conflict_steps:
        Prediction steps (1,...,N) at which the conflict exclusion is
        enforced.  None means terminal step N only, matching the workshop's
        T=t_min special case.
    yield_boundary:
        'geometric'       -> d <= d_max from vehicle/bike geometry.
        'workshop_center' -> d <= d_bike, reproducing the looser modified
                             constraint visible in the supplied notebook.
    terminal_stop:
        If True, additionally enforce v_N=0 for the yield strategy.
        The supplied workshop has this constraint commented out.
    discretization:
        'euler' reproduces the supplied notebook; 'zoh' is an explicitly
        marked modeling extension.
    """
    x0 = np.asarray(x0, float).reshape(2)
    strategy = str(strategy).lower()

    if strategy not in {"yield", "pass"}:
        raise ValueError("strategy must be 'yield' or 'pass'")

    N = int(N)
    if N < 1:
        raise ValueError("N must be positive")

    A, B = vehicle_matrices(Ts, discretization)
    Sx, Su = prediction_matrices(A, B, N)

    d_max, d_min = conflict_bounds(d_bike, l_car, w_bike)
    if yield_boundary == "geometric":
        d_yield = d_max
    elif yield_boundary == "workshop_center":
        d_yield = float(d_bike)
    else:
        raise ValueError(
            "yield_boundary must be 'geometric' or 'workshop_center'"
        )

    d_pass = d_min

    if conflict_steps is None:
        conflict_steps = (N,)
    else:
        conflict_steps = tuple(sorted(set(int(k) for k in conflict_steps)))

    if any(k < 1 or k > N for k in conflict_steps):
        raise ValueError("conflict_steps must lie in {1,...,N}")

    # Initial state is a real state and must satisfy the speed bounds too.
    if x0[1] < -1e-10 or x0[1] > v_max + 1e-10:
        return _empty_solution(
            strategy, x0,
            message="initial speed violates [0,v_max]",
            conflict_steps=conflict_steps,
            d_yield=d_yield,
            d_pass=d_pass,
            terminal_stop=terminal_stop,
            discretization=discretization,
        )

    # Cost:
    #   sum_{k=0}^{N-1} Ru a_k^2
    # + sum_{k=0}^{N-1} Qv (v_k-v_ref)^2
    # + Pv (v_N-v_ref)^2
    #
    # x0 is fixed, so its stage term is an optimization constant.
    q_blocks = []
    for k in range(1, N + 1):
        speed_weight = Pv if k == N else Qv
        q_blocks.append(np.diag([0.0, speed_weight]))

    Qbar = block_diag(*q_blocks)
    Rbar = Ru * np.eye(N)

    xref_stack = np.tile(np.array([0.0, v_ref]), N)
    base = Sx @ x0 - xref_stack

    H = 2.0 * (Su.T @ Qbar @ Su + Rbar)
    f = 2.0 * Su.T @ Qbar @ base

    Aubs = []
    bubs = []

    # Predicted speed constraints 0 <= v_k <= v_max, k=1,...,N.
    for k in range(1, N + 1):
        idx = 2 * (k - 1) + 1
        row = Su[idx]
        offset = Sx[idx] @ x0

        Aubs.append(row)
        bubs.append(v_max - offset)

        Aubs.append(-row)
        bubs.append(offset)

    # Conflict exclusion at selected prediction steps.
    for k in conflict_steps:
        idx = 2 * (k - 1)
        row = Su[idx]
        offset = Sx[idx] @ x0

        if strategy == "yield":
            # d_k <= d_yield
            Aubs.append(row)
            bubs.append(d_yield - offset)
        else:
            # d_k >= d_pass  <=>  -d_k <= -d_pass
            Aubs.append(-row)
            bubs.append(-d_pass + offset)

    A_eq = None
    b_eq = None

    if terminal_stop and strategy == "yield":
        # v_N = 0
        idx = 2 * (N - 1) + 1
        A_eq = np.atleast_2d(Su[idx])
        b_eq = np.array([-Sx[idx] @ x0])

    amax = float(mu) * float(g)

    res = solve_qp(
        H,
        f,
        A_ub=np.asarray(Aubs, float),
        b_ub=np.asarray(bubs, float),
        A_eq=A_eq,
        b_eq=b_eq,
        lb=-amax * np.ones(N),
        ub= amax * np.ones(N),
        x0=warm,
        maxiter=1000,
    )

    U = np.asarray(res.x, float).reshape(N, 1)
    Xpred = (Sx @ x0 + Su @ res.x).reshape(N, 2)
    X = np.vstack([x0, Xpred])

    objective = _full_cost(
        X, U,
        v_ref=v_ref,
        Qv=Qv,
        Ru=Ru,
        Pv=Pv,
    )

    return CollisionSolution(
        strategy=strategy,
        success=bool(res.success),
        objective=objective if res.success else np.inf,
        X=X,
        U=U,
        message=res.message,
        max_violation=float(res.max_violation),
        conflict_steps=tuple(conflict_steps),
        d_yield=float(d_yield),
        d_pass=float(d_pass),
        terminal_stop=bool(terminal_stop),
        discretization=discretization,
    )


def choose_collision_strategy(x0, **kwargs):
    """
    Enumerate the two logical strategies and choose the lower-cost feasible one.

    With exactly one binary decision s in {0,1}, this is mathematically
    equivalent to exhaustive enumeration of the mixed-integer alternatives:
        min{J_yield^*, J_pass^*}.
    """
    solutions = [
        solve_collision_strategy(x0, "yield", **kwargs),
        solve_collision_strategy(x0, "pass", **kwargs),
    ]

    feasible = [sol for sol in solutions if sol.success]

    if not feasible:
        return None, solutions

    return min(feasible, key=lambda sol: sol.objective), solutions


def strategy_grid(d_values, v_values, **kwargs):
    """
    Evaluate strategy feasibility and optimal choice on an initial-state grid.

    Returns integer arrays with codes:
        feasible_count: 0,1,2
        best_code: -1 infeasible, 0 yield, 1 pass
    """
    d_values = np.asarray(d_values, float)
    v_values = np.asarray(v_values, float)

    feasible_count = np.zeros((len(v_values), len(d_values)), dtype=int)
    best_code = -np.ones((len(v_values), len(d_values)), dtype=int)

    for iv, v0 in enumerate(v_values):
        for idd, d0 in enumerate(d_values):
            best, sols = choose_collision_strategy(
                np.array([d0, v0]),
                **kwargs,
            )

            feasible_count[iv, idd] = sum(sol.success for sol in sols)

            if best is not None:
                best_code[iv, idd] = 0 if best.strategy == "yield" else 1

    return feasible_count, best_code


def terminal_position_extrema_no_speed_limits(
    x0,
    *,
    N=4,
    Ts=1.0,
    mu=0.10,
    g=9.81,
):
    """
    Analytic terminal-position extrema for the *Euler* workshop model
    when speed constraints are inactive.

    d_N = d_0 + N Ts v_0
          + Ts^2 sum_{j=0}^{N-2} (N-1-j) a_j.

    The last acceleration a_{N-1} affects v_N but not d_N under the
    explicit Euler position update.
    """
    d0, v0 = np.asarray(x0, float).reshape(2)
    amax = float(mu) * float(g)

    leverage = Ts**2 * N * (N - 1) / 2.0
    nominal = d0 + N * Ts * v0

    return (
        float(nominal - amax * leverage),
        float(nominal + amax * leverage),
    )
