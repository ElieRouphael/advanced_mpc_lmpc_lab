
import numpy as np

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.polytopes import (
    HPolytope,
    closed_loop_admissible_set_scalar_input,
    maximal_invariant,
)


def _problem():
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    B = np.array([[0.0],
                  [1.0]])
    Q = np.eye(2)
    R = np.array([[10.0]])
    X = HPolytope.box([-5.0, -5.0], [5.0, 5.0])
    K, P, _ = dlqr(A, B, Q, R)
    XK = closed_loop_admissible_set_scalar_input(X, K, -0.5, 0.5)
    Xf, _ = maximal_invariant(A - B @ K, XK)
    return A, B, Q, R, X, K, P, Xf


def test_dare_terminal_decrease_identity():
    A, B, Q, R, _, K, P, _ = _problem()
    Acl = A - B @ K
    residual = Acl.T @ P @ Acl - P + Q + K.T @ R @ K
    assert np.linalg.norm(residual, ord=np.inf) < 1e-9


def test_terminal_set_respects_state_input_and_invariance():
    A, B, _, _, X, K, _, Xf = _problem()
    Acl = A - B @ K

    for v in Xf.vertices_2d():
        assert X.contains(v, tol=1e-7)
        u = float((-K @ v).item())
        assert -0.5 - 1e-7 <= u <= 0.5 + 1e-7
        assert Xf.contains(Acl @ v, tol=1e-6)


def test_terminal_mpc_is_recursively_feasible_for_demo_state():
    A, B, Q, R, _, _, P, Xf = _problem()

    ctrl = LinearMPC(
        A, B, Q, R, N=5, P=P,
        x_lower=np.array([-5.0, -5.0]),
        x_upper=np.array([5.0, 5.0]),
        u_lower=np.array([-0.5]),
        u_upper=np.array([0.5]),
        terminal_H=Xf.H,
        terminal_h=Xf.h,
    )

    x0 = np.array([4.5, 0.5])
    sol0 = ctrl.solve(x0)
    assert sol0.success

    x1 = A @ x0 + B @ sol0.U[0]
    sol1 = ctrl.solve(x1)
    assert sol1.success
