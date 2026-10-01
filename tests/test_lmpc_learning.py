import numpy as np

from advanced_mpc_lmpc.lmpc import (
    trajectory_cost_to_go,
    solve_ftocp_l1,
    convex_safe_value,
    LMPCController,
)


def _problem():
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    B = np.array([[0.0],
                  [1.0]])
    Q = np.eye(2)
    Q1 = 0.01 * Q
    R = np.array([[1.0]])
    return A, B, Q, Q1, R


def test_cost_to_go_recursion():
    A, B, Q, Q1, R = _problem()

    X = np.array([
        [2.0, 0.0],
        [2.0, -1.0],
        [1.0, -1.0],
        [0.0, 0.0],
    ])
    U = np.array([
        [-1.0],
        [ 0.0],
        [ 1.0],
    ])

    J = trajectory_cost_to_go(X, U, Q, R, Q1)

    assert np.isclose(J[-1], 0.0)
    for k in range(len(U)):
        h = X[k] @ Q @ X[k] + U[k] @ R @ U[k] + np.linalg.norm(Q1 @ X[k], 1)
        assert np.isclose(J[k], h + J[k+1])


def test_convex_safe_value_1d():
    S = np.array([[-1.0], [0.0], [1.0]])
    V = np.array([1.0, 0.0, 1.0])

    value, lam, success = convex_safe_value(
        np.array([0.5]),
        S,
        V,
    )

    assert success
    assert np.isclose(np.sum(lam), 1.0)
    assert np.all(lam >= -1e-10)
    assert np.allclose(S.T @ lam, [0.5])
    assert np.isclose(value, 0.5)


def test_source_ftocp_respects_box_constraints():
    A, B, Q, Q1, R = _problem()

    sol = solve_ftocp_l1(
        A, B, Q, R, Q1,
        N=3,
        x0=np.array([-15.0, 0.0]),
        x_lower=np.array([-15.0, -15.0]),
        x_upper=np.array([ 15.0,  15.0]),
        u_lower=np.array([-5.0]),
        u_upper=np.array([ 5.0]),
    )

    assert sol.success
    assert np.max(np.abs(sol.X)) <= 15.0 + 1e-6
    assert np.max(np.abs(sol.U)) <= 5.0 + 1e-6


def test_lmpc_terminal_state_is_barycentric_safe_state():
    A, B, Q, Q1, R = _problem()

    X = np.array([
        [-3.0,  0.0],
        [-3.0,  2.0],
        [-1.0,  1.0],
        [ 0.0,  0.0],
    ])
    U = np.array([
        [ 2.0],
        [-1.0],
        [-1.0],
    ])

    ctrl = LMPCController(
        A, B, Q, R,
        N=2,
        Q1=Q1,
        x_lower=np.array([-15.0, -15.0]),
        x_upper=np.array([ 15.0,  15.0]),
        u_lower=np.array([-5.0]),
        u_upper=np.array([ 5.0]),
    )
    ctrl.add_trajectory(X, U)

    sol = ctrl.solve(np.array([-3.0, 0.0]))

    assert sol.success
    assert np.isclose(np.sum(sol.lambdas), 1.0, atol=1e-6)
    assert np.min(sol.lambdas) >= -1e-7
    assert np.allclose(
        sol.X[-1],
        sol.safe_states.T @ sol.lambdas,
        atol=1e-6,
    )
