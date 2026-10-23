import numpy as np
import pytest

from advanced_mpc_lmpc.racing import (
    GlobalRacingModel,
    LMPC_INPUT_BOUNDS,
    LocalRacingModel,
    bicycle_lateral_forces,
    bicycle_spatial_rhs,
    build_demo_track,
    friction_circle_utilization,
    frenet_rates,
    kinematic_affine_linearization,
    lateral_speed_squared_cap,
    periodic_minimum_time_speed_profile,
    rollout_affine_predictive_model,
    scale_track_geometry,
    segment_time_from_speed_squared,
    simulate_pid_lap,
    track_global_position,
)


@pytest.fixture(scope="module")
def track():
    return build_demo_track(n=240, half_width=3.0)


@pytest.fixture(scope="module")
def rc_data():
    track = scale_track_geometry(build_demo_track(n=400), factor=0.08, half_width=0.4)
    laps = [simulate_pid_lap(track, seed=seed) for seed in range(3)]
    return track, laps


def test_demo_track_geometry(track):
    assert track.length == pytest.approx(252.333, abs=1e-3)
    assert np.allclose(track.ds, track.length / 240)
    i = int(np.argmax(np.abs(track.curvature)))
    assert track.s[i] == pytest.approx(99.882, abs=1e-3)
    assert track.curvature[i] == pytest.approx(0.12943, abs=1e-5)
    # closed loop: the heading turns by exactly one revolution
    step = track.heading[0] + 2 * np.pi - track.heading[-1]
    assert 0 < step < 0.1
    assert np.allclose(track.to_global(0.0), track.xy)


def test_track_global_position_matches_samples(track):
    xy = track_global_position(track, track.s[:10], 1.5)
    assert np.allclose(xy, track.to_global(1.5)[:10])


def test_frenet_rates_by_hand():
    r = frenet_rates(vx=12.0, vy=0.4, yaw_rate=0.15, epsi=0.05, kappa=0.02, ey=0.5)
    assert r["denominator"] == pytest.approx(0.99)
    assert r["tangential_velocity"] == pytest.approx(12 * np.cos(0.05) - 0.4 * np.sin(0.05))
    assert r["sdot"] == pytest.approx(r["tangential_velocity"] / 0.99)
    assert r["epsidot"] == pytest.approx(0.15 - 0.02 * r["sdot"])
    assert r["dt_ds"] * r["sdot"] == pytest.approx(1.0)


def test_bicycle_model_reference_values():
    f = bicycle_lateral_forces(vx=10.0, vy=0.25, yaw_rate=0.2, delta=0.08)
    assert f["alpha_f"] == pytest.approx(0.05250693, abs=1e-8)
    assert f["Fyf"] == pytest.approx(0.50910915, abs=1e-8)
    assert f["Fyr"] == pytest.approx(-0.21841526, abs=1e-8)
    rhs = bicycle_spatial_rhs(np.array([10.0, 0.25, 0.20, 0.04, 0.0, 0.30]), ax=1.0, delta=0.08, kappa=0.03)
    expected = [0.10220261, -0.18406337, 0.37534374, -0.01014427, 0.09927866, 0.06450069]
    assert np.allclose(rhs, expected, atol=1e-8)


def test_segment_time_is_exact_for_constant_acceleration():
    assert segment_time_from_speed_squared(10.0**2, 14.0**2, 30.0) == pytest.approx(2.5)


def test_speed_profile_is_feasible_and_tight(track):
    sol = periodic_minimum_time_speed_profile(track, mu=1.0, vmax=35.0, amin=-6.0, amax=3.0)
    assert sol["success"]
    assert sol["time"] == pytest.approx(17.0939, abs=1e-3)
    assert np.all(sol["a"] <= 3.0 + 1e-8) and np.all(sol["a"] >= -6.0 - 1e-8)
    assert np.all(sol["z"] <= sol["zmax"] + 1e-8)
    # the slowest point is the tightest corner, at its lateral-acceleration limit
    assert sol["v"].min() == pytest.approx(np.sqrt(9.81 / np.max(np.abs(track.curvature))))
    # largest feasible profile: lowering any bound or raising mu cannot make the lap slower
    faster = periodic_minimum_time_speed_profile(track, mu=1.2)
    assert faster["time"] < sol["time"]


def test_speed_cap_and_friction_diagnostic():
    cap = lateral_speed_squared_cap(np.array([0.0, 0.1]), mu=1.0, vmax=20.0)
    assert np.allclose(cap, [400.0, 98.1])
    usage = friction_circle_utilization(np.array([98.1]), np.array([0.0]), np.array([0.1]))
    assert usage[0] == pytest.approx(1.0)


def test_pid_laps_complete_and_respect_bounds(rc_data):
    track, laps = rc_data
    for X, U in laps:
        assert len(X) == len(U) + 1
        assert X[-1, 4] >= track.length and X[-2, 4] < track.length
        assert np.all(U >= LMPC_INPUT_BOUNDS[0]) and np.all(U <= LMPC_INPUT_BOUNDS[1])
        assert np.max(np.abs(X[:, 5])) < track.half_width


def test_kinematic_linearization_matches_finite_differences(rc_data):
    track, laps = rc_data
    x = laps[0][0][120]
    A, c, kappa = kinematic_affine_linearization(x, track, dt=0.1)

    def euler_step(z):
        """Euler step of (e_psi, s, e_y) with the curvature frozen at its value at x."""
        vx, vy, r, epsi, s, ey = z
        sdot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - kappa * ey)
        return np.array([epsi + 0.1 * (r - kappa * sdot), s + 0.1 * sdot,
                         ey + 0.1 * (vx * np.sin(epsi) + vy * np.cos(epsi))])

    f = euler_step
    # A x + c reproduces the Euler step exactly at the linearization point
    assert np.allclose(A @ x + c, f(x))

    eps = 1e-6
    J = np.column_stack([(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(6)])
    assert np.allclose(A, J, atol=1e-7)


def test_local_and_global_models(rc_data):
    track, laps = rc_data
    local = LocalRacingModel(track, dt=0.1, max_points_per_lap=7, bandwidth=5.0, ridge=1e-8)
    global_ = GlobalRacingModel(track, dt=0.1, ridge=1e-8)
    for X, U in laps[:2]:
        local.add_trajectory(X, U)
        global_.add_trajectory(X, U)

    X_val, U_val = laps[2]
    err_local, err_global = [], []
    for k in range(0, len(U_val), 5):
        x_next, diag = local.predict(X_val[k], U_val[k])
        assert not any(diag.support_fallback)
        assert all(len(d) <= 7 for d in diag.selected_distances)
        assert all(np.all(w > 0) for w in diag.selected_weights)
        err_local.append(x_next[:3] - X_val[k + 1, :3])
        err_global.append(global_.predict(X_val[k], U_val[k])[:3] - X_val[k + 1, :3])
    rmse_local = np.sqrt(np.mean(np.square(err_local)))
    rmse_global = np.sqrt(np.mean(np.square(err_global)))
    assert rmse_local < rmse_global

    # far from the data the local model falls back and says so
    x_far, u_far = np.array([4.0, 1.0, 4.0, 0.2, 5.0, 0.2]), np.array([0.5, 5.0])
    assert local.support_distance(x_far, u_far) > local.bandwidth
    assert all(local.fit(x_far, u_far)[3].support_fallback)

    X_pred, diags = rollout_affine_predictive_model(local, X_val[50], U_val[50:60])
    assert X_pred.shape == (11, 6) and len(diags) == 10
    assert np.all(np.isfinite(X_pred))
