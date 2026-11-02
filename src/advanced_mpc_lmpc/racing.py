"""Racing tools: track geometry, a minimum-time speed planner, a dynamic bicycle
model in Frenet coordinates, and learned prediction models for racing MPC.

Used by notebook 06 (minimum-time racing) and notebook 07 (data-driven local
models). The vehicle state and input are

    x = [vx, vy, r, e_psi, s, e_y],     u = [delta, a],

with longitudinal and lateral velocity in the body frame, yaw rate, heading
error relative to the centerline, arc length along the centerline, signed
lateral deviation (positive to the left), steering angle and longitudinal
acceleration.

The local regression in ``LocalRacingModel`` follows the method of
U. Rosolia and F. Borrelli, "Learning how to autonomously race a car: a
predictive control approach", IEEE Transactions on Control Systems Technology,
28(6), 2020.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


# ---------------------------------------------------------------- track geometry

@dataclass
class Track:
    """A closed centerline sampled uniformly in arc length."""

    s: np.ndarray          # arc length of each sample [m], starting at 0
    xy: np.ndarray         # centerline points, shape (n, 2)
    heading: np.ndarray    # tangent angle of the centerline [rad], unwrapped
    curvature: np.ndarray  # signed curvature [1/m], positive for left turns
    length: float          # length of the closed centerline [m]
    half_width: float      # half of the drivable width [m]

    @property
    def ds(self) -> np.ndarray:
        """Length of segment i, from sample i to sample i+1; the last segment closes the loop."""
        return np.diff(np.r_[self.s, self.length])

    def wrap(self, s):
        """Arc length reduced to one lap, [0, length)."""
        return np.mod(np.asarray(s, float), self.length)

    def curvature_at(self, s):
        """Curvature at arbitrary arc length, by periodic linear interpolation."""
        grid = np.r_[self.s, self.length]
        return np.interp(self.wrap(s), grid, np.r_[self.curvature, self.curvature[0]])

    def normal(self) -> np.ndarray:
        """Unit left normal at every sample: positive e_y lies to the left of the direction of travel."""
        return np.c_[-np.sin(self.heading), np.cos(self.heading)]

    def to_global(self, ey) -> np.ndarray:
        """Points at lateral offset ``ey`` from every centerline sample, shape (n, 2)."""
        ey = np.broadcast_to(np.asarray(ey, float), self.s.shape)
        return self.xy + ey[:, None] * self.normal()


def build_demo_track(n: int = 400, half_width: float = 3.0, dense: int = 200_000) -> Track:
    """Smooth closed demonstration track, sampled at ``n`` points uniform in arc length.

    The centerline is the closed curve

        x(t) = 45 cos t + 5 cos 2t,    y(t) = 30 sin t + 6 sin 3t,    t in [0, 2 pi),

    driven counter-clockwise from (50, 0). It is about 252.3 m long, with two
    tight corners (radius about 7.7 m), two sweeping bends and two slight reverse
    kinks on the long sides. Curvature and heading are evaluated from the exact
    derivatives of the curve; arc length is integrated on a dense parameter grid.
    """
    t = np.linspace(0.0, 2.0 * np.pi, dense + 1)
    speed = np.hypot(*_demo_curve_d1(t))
    s_dense = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))]
    length = float(s_dense[-1])

    s = np.arange(n) * length / n
    tq = np.interp(s, s_dense, t)
    half = n // 2
    if n % 2 == 0:
        # The curve is symmetric about the x-axis: sample it symmetrically so the two
        # mirror-image corners get identical samples.
        tq[half + 1:] = 2.0 * np.pi - tq[1:half][::-1]
    x, y = _demo_curve(tq)
    dx, dy = _demo_curve_d1(tq)
    ddx, ddy = _demo_curve_d2(tq)
    curvature = (dx * ddy - dy * ddx) / np.hypot(dx, dy) ** 3
    if n % 2 == 0:
        curvature[half + 1:] = curvature[1:half][::-1]
    heading = np.unwrap(np.arctan2(dy, dx))
    return Track(s=s, xy=np.c_[x, y], heading=heading, curvature=curvature,
                 length=length, half_width=float(half_width))


def _demo_curve(t):
    return 45 * np.cos(t) + 5 * np.cos(2 * t), 30 * np.sin(t) + 6 * np.sin(3 * t)


def _demo_curve_d1(t):
    return -45 * np.sin(t) - 10 * np.sin(2 * t), 30 * np.cos(t) + 18 * np.cos(3 * t)


def _demo_curve_d2(t):
    return -45 * np.cos(t) - 20 * np.cos(2 * t), -30 * np.sin(t) - 54 * np.sin(3 * t)


def scale_track_geometry(track: Track, factor: float, half_width: float | None = None) -> Track:
    """The same track scaled by ``factor`` (lengths multiply, curvature divides)."""
    return Track(s=track.s * factor, xy=track.xy * factor, heading=track.heading.copy(),
                 curvature=track.curvature / factor, length=track.length * factor,
                 half_width=float(track.half_width * factor if half_width is None else half_width))


def track_global_position(track: Track, s, ey) -> np.ndarray:
    """Global (X, Y) positions of Frenet points (s, e_y), shape (m, 2)."""
    s = track.wrap(np.atleast_1d(s))
    ey = np.broadcast_to(np.asarray(ey, float), s.shape)
    grid = np.r_[track.s, track.length]

    def periodic(values):
        return np.interp(s, grid, np.r_[values, values[0]])

    x, y = periodic(track.xy[:, 0]), periodic(track.xy[:, 1])
    cos_h, sin_h = periodic(np.cos(track.heading)), periodic(np.sin(track.heading))
    norm = np.hypot(cos_h, sin_h)
    return np.c_[x - ey * sin_h / norm, y + ey * cos_h / norm]


# ---------------------------------------------------------------- Frenet kinematics

def frenet_rates(vx, vy, yaw_rate, epsi, kappa, ey) -> dict:
    """Time and space derivatives of the Frenet coordinates for a planar vehicle.

    s_dot = (vx cos e_psi - vy sin e_psi) / (1 - kappa e_y),
    e_y_dot = vx sin e_psi + vy cos e_psi,
    e_psi_dot = r - kappa s_dot,
    and the spatial rates d(.)/ds = (.)_dot / s_dot.
    """
    denominator = 1.0 - kappa * ey
    tangential = vx * math.cos(epsi) - vy * math.sin(epsi)
    sdot = tangential / denominator
    eydot = vx * math.sin(epsi) + vy * math.cos(epsi)
    epsidot = yaw_rate - kappa * sdot
    return {
        "sdot": sdot,
        "eydot": eydot,
        "epsidot": epsidot,
        "dt_ds": 1.0 / sdot,
        "dey_ds": eydot / sdot,
        "depsi_ds": epsidot / sdot,
        "denominator": denominator,
        "tangential_velocity": tangential,
    }


# ---------------------------------------------------------------- dynamic bicycle model

@dataclass(frozen=True)
class VehicleParams:
    """Dynamic bicycle model of a 1:10-scale RC car (the BARC platform).

    Each axle's lateral force follows the simplified Pacejka formula
    F = D sin(C arctan(B alpha)), with peak force D = 0.8 m g / 2.
    """

    m: float = 1.98     # mass [kg]
    lf: float = 0.125   # centre of gravity to front axle [m]
    lr: float = 0.125   # centre of gravity to rear axle [m]
    Iz: float = 0.024   # yaw moment of inertia [kg m^2]
    B: float = 1.0      # tire stiffness factor
    C: float = 1.25     # tire shape factor
    peak_friction: float = 0.8  # peak lateral force / normal load on each axle

    @property
    def D(self) -> float:
        return self.peak_friction * self.m * G / 2.0


BARC = VehicleParams()

# Input bounds used by the Learning MPC racing controller: |delta| <= 0.5 rad, |a| <= 10 m/s^2.
LMPC_INPUT_BOUNDS = np.array([[-0.5, -10.0], [0.5, 10.0]])


def bicycle_lateral_forces(vx, vy, yaw_rate, delta, params: VehicleParams = BARC) -> dict:
    """Slip angles and lateral tire forces of the dynamic bicycle model."""
    alpha_f = delta - math.atan2(vy + params.lf * yaw_rate, vx)
    alpha_r = -math.atan2(vy - params.lr * yaw_rate, vx)
    Fyf = params.D * math.sin(params.C * math.atan(params.B * alpha_f))
    Fyr = params.D * math.sin(params.C * math.atan(params.B * alpha_r))
    return {"alpha_f": alpha_f, "alpha_r": alpha_r, "Fyf": Fyf, "Fyr": Fyr}


def bicycle_dynamics(x, u, kappa, params: VehicleParams = BARC) -> np.ndarray:
    """Time derivative of x = [vx, vy, r, e_psi, s, e_y] for input u = [delta, a]."""
    vx, vy, r, epsi, _, ey = (float(v) for v in x)
    delta, a = float(u[0]), float(u[1])
    f = bicycle_lateral_forces(vx, vy, r, delta, params)
    Fyf, Fyr = f["Fyf"], f["Fyr"]
    sdot = (vx * math.cos(epsi) - vy * math.sin(epsi)) / (1.0 - kappa * ey)
    return np.array([
        a - Fyf * math.sin(delta) / params.m + r * vy,
        (Fyf * math.cos(delta) + Fyr) / params.m - r * vx,
        (params.lf * Fyf * math.cos(delta) - params.lr * Fyr) / params.Iz,
        r - kappa * sdot,
        sdot,
        vx * math.sin(epsi) + vy * math.cos(epsi),
    ])


def bicycle_spatial_rhs(state, ax, delta, kappa, params: VehicleParams = BARC) -> np.ndarray:
    """Spatial dynamics d(state)/ds of the bicycle model, state = [vx, vy, r, e_psi, t, e_y].

    Arc length replaces time as the independent variable: every time derivative
    is divided by s_dot, and elapsed time becomes a state with dt/ds = 1/s_dot.
    """
    vx, vy, r, epsi, _, ey = (float(v) for v in state)
    xdot = bicycle_dynamics([vx, vy, r, epsi, 0.0, ey], [delta, ax], kappa, params)
    sdot = xdot[4]
    return np.array([xdot[0], xdot[1], xdot[2], xdot[3], 1.0, xdot[5]]) / sdot


# ---------------------------------------------------------------- minimum-time speed planning

def lateral_speed_squared_cap(curvature, mu=1.0, g=G, vmax=35.0, eps=1e-9) -> np.ndarray:
    """Pointwise bound on z = v^2: lateral acceleration z |kappa| <= mu g, and v <= vmax."""
    curvature = np.abs(np.asarray(curvature, float))
    return np.minimum(vmax ** 2, mu * g / np.maximum(curvature, eps))


def segment_time_from_speed_squared(z0, z1, ds):
    """Time to cover ds at constant acceleration between speeds sqrt(z0) and sqrt(z1)."""
    return 2.0 * ds / (np.sqrt(z0) + np.sqrt(z1))


def periodic_minimum_time_speed_profile(track: Track, mu=1.0, g=G, vmax=35.0,
                                        amin=-6.0, amax=3.0, tol=1e-9, max_iter=100) -> dict:
    """Minimum-time speed profile on a closed track, in the spatial domain.

    Decision variable z_i = v_i^2 at every sample. Over segment i of length ds_i,
    a constant acceleration a_i gives z_{i+1} = z_i + 2 a_i ds_i, so the bounds
    amin <= a_i <= amax are linear in z:

        z_{i+1} <= z_i + 2 amax ds_i        (acceleration)
        z_i     <= z_{i+1} - 2 amin ds_i    (braking)

    together with z_i <= cap_i from ``lateral_speed_squared_cap``. Lap time
    decreases in every z_i, so the minimum-time profile is the largest feasible
    z. It is found by repeated forward (acceleration) and backward (braking)
    passes around the loop, starting from the cap, until nothing changes.
    """
    ds = track.ds
    n = len(ds)
    zmax = lateral_speed_squared_cap(track.curvature, mu=mu, g=g, vmax=vmax)
    z = zmax.copy()
    converged = False
    for iteration in range(1, max_iter + 1):
        z_previous = z.copy()
        for i in range(n):                      # forward: acceleration limit
            j = (i + 1) % n
            z[j] = min(z[j], z[i] + 2.0 * amax * ds[i])
        for i in range(n - 1, -1, -1):          # backward: braking limit
            j = (i + 1) % n
            z[i] = min(z[i], z[j] - 2.0 * amin * ds[i])
        if np.max(np.abs(z - z_previous)) <= tol:
            converged = True
            break
    z_next = np.roll(z, -1)
    a = (z_next - z) / (2.0 * ds)
    segment_time = segment_time_from_speed_squared(z, z_next, ds)
    feasible = (np.all(z <= zmax + 1e-8) and np.all(a <= amax + 1e-8)
                and np.all(a >= amin - 1e-8) and np.all(z > 0))
    return {
        "success": bool(converged and feasible),
        "iterations": iteration,
        "time": float(np.sum(segment_time)),
        "v": np.sqrt(z),
        "a": a,
        "z": z,
        "zmax": zmax,
        "segment_time": segment_time,
    }


def friction_circle_utilization(z, a, curvature, mu=1.0, g=G) -> np.ndarray:
    """sqrt(a_x^2 + a_y^2) / (mu g) with lateral acceleration a_y = z kappa.

    The speed planner bounds a_x and a_y separately; values above 1 show where the
    combined (friction-circle) demand exceeds what the tires can deliver.
    """
    a_lat = np.asarray(z, float) * np.asarray(curvature, float)
    return np.hypot(np.asarray(a, float), a_lat) / (mu * g)


# ---------------------------------------------------------------- exploration data

def pid_control(x, target_speed=0.8, rng=None, steer_noise=0.25, accel_noise=0.10) -> np.ndarray:
    """Noisy path-following controller used to collect identification laps.

    Steering corrects lateral and heading errors, acceleration tracks the target
    speed, and clipped Gaussian noise excites the dynamics so the recorded data
    cover more of the state-input space.
    """
    rng = np.random.default_rng() if rng is None else rng
    steer_dither = float(np.clip(steer_noise * rng.standard_normal(), -0.9, 0.9))
    accel_dither = float(np.clip(accel_noise * rng.standard_normal(), -0.2, 0.2))
    delta = -0.6 * x[5] - 0.9 * x[3] + steer_dither
    a = 1.5 * (target_speed - x[0]) + accel_dither
    return np.array([delta, a])


def simulate_pid_lap(track: Track, seed=0, target_speed=0.8, dt=0.1, inner_dt=0.001,
                     x0=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0), enforce_lmpc_bounds=True,
                     params: VehicleParams = BARC, max_steps=2000):
    """One lap of the bicycle model under ``pid_control``.

    The input is held for ``dt`` while the dynamics are integrated with explicit
    Euler steps of ``inner_dt``. Returns states X (T+1, 6) and inputs U (T, 2);
    the lap ends at the first sample with s >= track.length.
    """
    rng = np.random.default_rng(seed)
    n_inner = int(round(dt / inner_dt))
    X = [np.asarray(x0, float)]
    U = []
    while X[-1][4] < track.length:
        if len(U) >= max_steps:
            raise RuntimeError("the lap did not finish within max_steps")
        u = pid_control(X[-1], target_speed, rng)
        if enforce_lmpc_bounds:
            u = np.clip(u, LMPC_INPUT_BOUNDS[0], LMPC_INPUT_BOUNDS[1])
        x = X[-1].copy()
        for _ in range(n_inner):
            x = x + inner_dt * bicycle_dynamics(x, u, float(track.curvature_at(x[4])), params)
        U.append(u)
        X.append(x)
    return np.array(X), np.array(U)


# ---------------------------------------------------------------- learned prediction models

def kinematic_affine_linearization(x, track: Track, dt=0.1):
    """Exact affine linearization of the Euler-discretized Frenet kinematics at x.

    Rows for [e_psi, s, e_y] at the next step: x_kin+ ~ A_kin x + c_kin. The local
    curvature is treated as constant over the step, so the derivative of kappa with
    respect to s is neglected. Returns (A_kin (3, 6), c_kin (3,), kappa).
    """
    vx, vy, r, epsi, s, ey = (float(v) for v in x)
    kappa = float(track.curvature_at(s))
    den = 1.0 - kappa * ey
    c, sn = math.cos(epsi), math.sin(epsi)
    tangential = vx * c - vy * sn
    d_tangential_depsi = -vx * sn - vy * c
    sdot = tangential / den

    A = np.zeros((3, 6))
    # e_psi+ = e_psi + dt (r - kappa s_dot)
    A[0] = [-dt * kappa * c / den, dt * kappa * sn / den, dt,
            1.0 - dt * kappa * d_tangential_depsi / den, 0.0, -dt * kappa ** 2 * tangential / den ** 2]
    # s+ = s + dt s_dot
    A[1] = [dt * c / den, -dt * sn / den, 0.0, dt * d_tangential_depsi / den, 1.0,
            dt * kappa * tangential / den ** 2]
    # e_y+ = e_y + dt (vx sin e_psi + vy cos e_psi)
    A[2] = [dt * sn, dt * c, 0.0, dt * tangential, 0.0, 1.0]

    f = np.array([epsi + dt * (r - kappa * sdot), s + dt * sdot, ey + dt * (vx * sn + vy * c)])
    return A, f - A @ np.asarray(x, float), kappa


@dataclass
class FitDiagnostics:
    """What a model fit used, for inspection in notebooks."""

    curvature: float
    selected_indices: list
    selected_distances: list
    selected_weights: list
    normal_condition_numbers: list
    support_fallback: list


# Structure of the dynamic rows: vx depends on (vx, vy, r) and a; vy and r depend on (vx, vy, r) and delta.
_LONGITUDINAL_INPUT = 1
_LATERAL_INPUT = 0


def _weighted_ridge(Phi, Y, w, ridge):
    """Solve min sum_k w_k ||Phi_k theta - Y_k||^2 + ridge ||theta||^2; returns (theta, cond(normal matrix))."""
    weighted = Phi * w[:, None]
    normal = Phi.T @ weighted + ridge * np.eye(Phi.shape[1])
    theta = np.linalg.solve(normal, weighted.T @ Y)
    return theta, float(np.linalg.cond(normal))


def _structured_dynamic_fit(X, U, Xnext, w, ridge):
    """Weighted fit of the rows for vx, vy and r. Returns (A (3, 6), B (3, 2), c (3,), condition numbers)."""
    ones = np.ones((len(X), 1))
    A = np.zeros((3, 6))
    B = np.zeros((3, 2))
    c = np.zeros(3)

    Phi_vx = np.hstack([X[:, :3], U[:, [_LONGITUDINAL_INPUT]], ones])
    theta_vx, cond_vx = _weighted_ridge(Phi_vx, Xnext[:, 0], w, ridge)
    A[0, :3], B[0, _LONGITUDINAL_INPUT], c[0] = theta_vx[:3], theta_vx[3], theta_vx[4]

    Phi_lat = np.hstack([X[:, :3], U[:, [_LATERAL_INPUT]], ones])
    theta_lat, cond_lat = _weighted_ridge(Phi_lat, Xnext[:, 1:3], w, ridge)
    A[1:3, :3] = theta_lat[:3].T
    B[1:3, _LATERAL_INPUT] = theta_lat[3]
    c[1:3] = theta_lat[4]
    return A, B, c, [cond_vx, cond_lat]


def _assemble(A_dyn, B_dyn, c_dyn, A_kin, c_kin):
    A = np.vstack([A_dyn, A_kin])
    B = np.vstack([B_dyn, np.zeros((3, 2))])
    return A, B, np.r_[c_dyn, c_kin]


class _RacingData:
    """Stored laps as transitions (x_k, u_k) -> x_{k+1}."""

    def __init__(self):
        self.laps = []   # list of (X_k, U_k, X_{k+1})

    def add_trajectory(self, X, U):
        X = np.asarray(X, float)
        U = np.asarray(U, float)
        if len(X) != len(U) + 1:
            raise ValueError("X must have one more row than U")
        self.laps.append((X[:-1], U, X[1:]))


class LocalRacingModel(_RacingData):
    """Local linear regression around the query point, one model per query.

    For a query (x, u), each stored lap contributes its transitions closest to
    (vx, vy, r, delta, a) in the scaled 1-norm

        d = sum_i w_i |z_i - z_query,i|,   w = (0.1, 1, 1, 1, 1),

    keeping at most ``max_points_per_lap`` of them inside the kernel bandwidth h,
    weighted by the Epanechnikov kernel K(d) = 0.75 (1 - (d / h)^2). The dynamic
    rows are fitted by weighted ridge least squares with the structure of the
    bicycle model; the kinematic rows come from ``kinematic_affine_linearization``.

    If a lap has no transition inside the bandwidth, its nearest
    ``max_points_per_lap`` transitions are used with the bandwidth widened to just
    cover them, and the lap is flagged in ``support_fallback``.
    """

    def __init__(self, track: Track, dt=0.1, max_points_per_lap=7, bandwidth=5.0, ridge=1e-8,
                 scaling=(0.1, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.track = track
        self.dt = dt
        self.max_points_per_lap = int(max_points_per_lap)
        self.bandwidth = float(bandwidth)
        self.ridge = float(ridge)
        self.scaling = np.asarray(scaling, float)

    def _lap_distances(self, x, u):
        query = np.r_[np.asarray(x, float)[:3], np.asarray(u, float)]
        return [np.abs(np.hstack([Xk[:, :3], Uk]) - query) @ self.scaling for Xk, Uk, _ in self.laps]

    def support_distance(self, x, u) -> float:
        """Distance from (x, u) to the nearest stored transition, in the regression metric."""
        return float(min(d.min() for d in self._lap_distances(x, u)))

    def fit(self, x, u):
        """Affine model x+ ~ A x + B u + c valid near (x, u). Returns (A, B, c, diagnostics)."""
        if not self.laps:
            raise RuntimeError("add at least one trajectory before fitting")
        h, m = self.bandwidth, self.max_points_per_lap
        rows_X, rows_U, rows_next, weights = [], [], [], []
        indices, distances, kernel_weights, fallback = [], [], [], []
        for (Xk, Uk, Xn), d in zip(self.laps, self._lap_distances(x, u)):
            order = np.argsort(d, kind="stable")
            inside = order[d[order] < h]
            if len(inside):
                idx, h_lap, flag = inside[:m], h, False
            else:
                idx = order[:m]
                h_lap, flag = 1.05 * float(d[idx].max()), True
            w = 0.75 * (1.0 - (d[idx] / h_lap) ** 2)
            rows_X.append(Xk[idx]); rows_U.append(Uk[idx]); rows_next.append(Xn[idx]); weights.append(w)
            indices.append(idx); distances.append(d[idx]); kernel_weights.append(w); fallback.append(flag)

        A_dyn, B_dyn, c_dyn, cond = _structured_dynamic_fit(
            np.vstack(rows_X), np.vstack(rows_U), np.vstack(rows_next), np.concatenate(weights), self.ridge)
        A_kin, c_kin, kappa = kinematic_affine_linearization(x, self.track, self.dt)
        A, B, c = _assemble(A_dyn, B_dyn, c_dyn, A_kin, c_kin)
        return A, B, c, FitDiagnostics(kappa, indices, distances, kernel_weights, cond, fallback)

    def predict(self, x, u):
        """One-step prediction with the local model at (x, u). Returns (x_next, diagnostics)."""
        A, B, c, diag = self.fit(x, u)
        return A @ np.asarray(x, float) + B @ np.asarray(u, float) + c, diag


class GlobalRacingModel(_RacingData):
    """One affine model of the dynamic rows fitted to all stored transitions.

    Same structure and kinematic rows as ``LocalRacingModel``, but every
    transition gets the same weight, wherever it was recorded.
    """

    def __init__(self, track: Track, dt=0.1, ridge=1e-8):
        super().__init__()
        self.track = track
        self.dt = dt
        self.ridge = float(ridge)
        self._cache = None

    def add_trajectory(self, X, U):
        super().add_trajectory(X, U)
        self._cache = None

    def _dynamic_rows(self):
        if self._cache is None:
            if not self.laps:
                raise RuntimeError("add at least one trajectory before fitting")
            X = np.vstack([lap[0] for lap in self.laps])
            U = np.vstack([lap[1] for lap in self.laps])
            Xn = np.vstack([lap[2] for lap in self.laps])
            self._cache = _structured_dynamic_fit(X, U, Xn, np.ones(len(X)), self.ridge)
        return self._cache

    def fit(self, x, u):
        """Affine model at (x, u): global dynamic rows, kinematic rows linearized at x."""
        A_dyn, B_dyn, c_dyn, cond = self._dynamic_rows()
        A_kin, c_kin, kappa = kinematic_affine_linearization(x, self.track, self.dt)
        A, B, c = _assemble(A_dyn, B_dyn, c_dyn, A_kin, c_kin)
        return A, B, c, FitDiagnostics(kappa, [], [], [], list(cond), [False] * len(self.laps))

    def predict(self, x, u):
        """One-step prediction x_next."""
        A, B, c, _ = self.fit(x, u)
        return A @ np.asarray(x, float) + B @ np.asarray(u, float) + c


def rollout_affine_predictive_model(model, x0, U):
    """Open-loop prediction over a horizon, re-fitting the affine model at every step.

    Returns predicted states (H+1, 6) and the list of fit diagnostics. This is the
    time-varying (LTV) prediction an MPC would build from the learned model.
    """
    X = [np.asarray(x0, float)]
    diagnostics = []
    for u in np.asarray(U, float):
        A, B, c, diag = model.fit(X[-1], u)
        X.append(A @ X[-1] + B @ u + c)
        diagnostics.append(diag)
    return np.array(X), diagnostics
