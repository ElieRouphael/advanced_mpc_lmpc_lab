import numpy as np
from scipy.linalg import solve_discrete_are


def prediction_matrices(A: np.ndarray, B: np.ndarray, N: int):
    """Return Sx, Su such that [x1;...;xN] = Sx x0 + Su [u0;...;uN-1]."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = B.shape
    Sx = np.zeros((N * n, n))
    Su = np.zeros((N * n, N * m))
    for i in range(N):
        Sx[i*n:(i+1)*n] = np.linalg.matrix_power(A, i + 1)
        for j in range(i + 1):
            Su[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i - j) @ B
    return Sx, Su


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Discrete-time infinite-horizon LQR with convention u = -Kx."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig = np.linalg.eigvals(A - B @ K)
    return K, P, eig


def rollout(A, B, x0, U):
    """Roll out an LTI system for an input sequence U with shape (N,m)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    x = np.asarray(x0, float).copy()
    xs = [x.copy()]
    for u in np.atleast_2d(U):
        u = np.asarray(u, float).reshape(-1)
        x = A @ x + B @ u
        xs.append(x.copy())
    return np.asarray(xs)
