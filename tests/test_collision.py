import numpy as np

from advanced_mpc_lmpc.collision import (
    conflict_bounds,
    vehicle_matrices,
    solve_collision_strategy,
    choose_collision_strategy,
    terminal_position_extrema_no_speed_limits,
)


def test_geometric_conflict_bounds():
    dmax, dmin = conflict_bounds(
        d_bike=100.0,
        l_car=4.5,
        w_bike=8.0,
    )
    assert np.isclose(dmax, 93.75)
    assert np.isclose(dmin, 106.25)


def test_euler_matrices():
    A, B = vehicle_matrices(Ts=1.0, discretization="euler")
    assert np.allclose(A, [[1.0, 1.0], [0.0, 1.0]])
    assert np.allclose(B, [[0.0], [1.0]])


def test_source_like_initial_condition_yields_but_cannot_pass():
    x0 = np.array([60.0, 5.0])
    y = solve_collision_strategy(x0, "yield")
    p = solve_collision_strategy(x0, "pass")

    assert y.success
    assert not p.success
    assert y.X[-1, 0] <= y.d_yield + 1e-6


def test_true_stop_is_stricter_than_yield_for_demo_state():
    x0 = np.array([60.0, 5.0])

    loose = solve_collision_strategy(
        x0, "yield",
        terminal_stop=False,
    )
    stopped = solve_collision_strategy(
        x0, "yield",
        terminal_stop=True,
    )

    assert loose.success
    assert not stopped.success


def test_enumeration_returns_lower_cost_feasible_mode():
    x0 = np.array([30.0, 17.0])

    best, sols = choose_collision_strategy(
        x0,
        mu=0.15,
        Ru=0.1,
    )

    feasible = [s for s in sols if s.success]
    assert len(feasible) == 2
    assert best.success
    assert np.isclose(best.objective, min(s.objective for s in feasible))


def test_analytic_euler_terminal_extrema_match_saturated_rollouts():
    x0 = np.array([20.0, 8.0])
    N = 4
    Ts = 1.0
    mu = 0.10
    g = 9.81
    amax = mu * g

    d_lo, d_hi = terminal_position_extrema_no_speed_limits(
        x0, N=N, Ts=Ts, mu=mu, g=g
    )

    A, B = vehicle_matrices(Ts, "euler")

    for sign, expected in [(-1.0, d_lo), (1.0, d_hi)]:
        x = x0.copy()
        for _ in range(N):
            x = A @ x + B[:, 0] * (sign * amax)
        assert np.isclose(x[0], expected)
