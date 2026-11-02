
import numpy as np

from advanced_mpc_lmpc.polytopes import (
    HPolytope,
    affine_image_2d,
    minkowski_sum_2d,
    n_step_controllable_sets,
    reachable_set_scalar_input_2d,
    closed_loop_admissible_set_scalar_input,
    predecessor_scalar_input,
)


def test_redundancy_removal_keeps_one_copy_of_duplicate_rows():
    X = HPolytope.box([-4.0, -3.0], [4.0, 3.0])
    XX = X.intersect(X)
    assert XX.H.shape[0] == 4
    assert XX.approximately_equal(X)
    assert not XX.contains([5.0, 0.0])


def test_n_step_controllable_sets_start_from_target():
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5], [1.0]])
    X = HPolytope.box([-4.0, -3.0], [4.0, 3.0])
    sets = n_step_controllable_sets(A, B, X, X, -1.0, 1.0, 2)
    assert sets[0].approximately_equal(X)
    # (4, 3) is in X but cannot be kept in X: every admissible input leaves it.
    assert X.contains([4.0, 3.0])
    assert not sets[1].contains([4.0, 3.0])


def test_minkowski_sum_boxes():
    P = HPolytope.box([-1.0, -2.0], [1.0, 2.0])
    Q = HPolytope.box([-0.5, -1.0], [0.5, 1.0])
    S = minkowski_sum_2d(P, Q)
    for v in np.array([
        [-1.5, -3.0],
        [ 1.5, -3.0],
        [ 1.5,  3.0],
        [-1.5,  3.0],
    ]):
        assert S.contains(v, tol=1e-7)


def test_affine_image_scaling():
    P = HPolytope.box([-1.0, -1.0], [1.0, 1.0])
    M = np.array([[2.0, 0.0], [0.0, 0.5]])
    Y = affine_image_2d(P, M)
    assert Y.contains([2.0, 0.5])
    assert Y.contains([-2.0, -0.5])
    assert not Y.contains([2.1, 0.0])


def test_forward_and_backward_sets_nonempty():
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5], [1.0]])
    S = HPolytope.box([-0.4, -0.4], [0.4, 0.4])

    Post = reachable_set_scalar_input_2d(S, A, B, -1.0, 1.0)
    Pre = predecessor_scalar_input(S, A, B, -1.0, 1.0)

    assert len(Post.vertices_2d()) >= 3
    assert len(Pre.vertices_2d()) >= 3


def test_closed_loop_admissible_set_respects_input():
    X = HPolytope.box([-4.0, -3.0], [4.0, 3.0])
    K = np.array([[0.4, 0.7]])
    XK = closed_loop_admissible_set_scalar_input(X, K, -1.0, 1.0)

    for v in XK.vertices_2d():
        u = float((-K @ v).item())
        assert u <= 1.0 + 1e-7
        assert u >= -1.0 - 1e-7
