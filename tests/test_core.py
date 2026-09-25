from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dynamic_icp.core import (
    Config, Scan, cluster_dynamics, deskew_scan, displacement_residual, dynamic_mask, estimate_ego_velocity, predict_source, register,
    reconstruct_velocities, residuals_and_jacobians, tukey_weights,
)


def test_jacobians_and_translation_invariance():
    rng = np.random.default_rng(5)
    p, q, n, us, ut = [rng.normal(size=(40, 3)) for _ in range(5)]
    n /= np.linalg.norm(n, axis=1)[:, None]
    us /= np.linalg.norm(us, axis=1)[:, None]
    ut /= np.linalg.norm(ut, axis=1)[:, None]
    s, t = rng.normal(size=(2, 40))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    T[:3, 3] = [1, 2, 3]
    args = (p, us, s, q, n, ut, t)
    rg, rv, Jg, Jv = residuals_and_jacobians(*args, T)
    for column in range(6):
        delta = np.zeros(6)
        delta[column] = 1e-7
        new_T = T.copy()
        new_T[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ T[:3, :3]
        new_T[:3, 3] += delta[3:]
        g, v, _, _ = residuals_and_jacobians(*args, new_T)
        np.testing.assert_allclose((g - rg) / 1e-7, Jg[:, column], atol=1e-6)
        np.testing.assert_allclose((v - rv) / 1e-7, Jv[:, column], atol=1e-6)
    T[:3, 3] += [100, -500, 800]
    np.testing.assert_array_equal(residuals_and_jacobians(*args, T)[1], rv)
    np.testing.assert_array_equal(Jv[:, 3:], 0)


def test_displacement_prior_jacobian():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    T[:3, 3] = [0.4, 0.5, -0.2]
    d = np.array([0.3, -0.1, 0.05])
    r, J = displacement_residual(T, d)
    for column in range(6):
        delta = np.eye(6)[column] * 1e-7
        other = T.copy()
        other[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ T[:3, :3]
        other[:3, 3] += delta[3:]
        np.testing.assert_allclose((displacement_residual(other, d)[0] - r) / 1e-7,
                                   J[:, column], atol=1e-7)


def test_deskew_recovers_static_points_and_rotates_only_los():
    rng = np.random.default_rng(78)
    truth = rng.normal(size=(100, 3)) + [10, 0, 0]
    times = np.linspace(0, 0.1, 100)
    velocity, omega = np.array([6, 1, 0.1]), np.array([0, 0, 0.3])
    rotations = Rotation.from_rotvec(times[:, None] * omega)
    raw = rotations.inv().apply(truth - times[:, None] * velocity)
    scan = Scan(raw, np.linspace(-5, -1, 100), time_offsets=times)
    corrected = deskew_scan(scan, velocity, omega)
    np.testing.assert_allclose(corrected.points, truth, atol=1e-12)
    np.testing.assert_allclose(corrected.los, rotations.apply(scan.los), atol=1e-12)
    np.testing.assert_array_equal(corrected.doppler, scan.doppler)
    np.testing.assert_array_equal(scan.time_offsets, times)
    with pytest.raises(ValueError, match="requires"):
        deskew_scan(Scan(raw, scan.doppler), velocity, omega)


def test_robust_ego_3d_and_range_filter():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(3000, 3)) * 10
    U = points / np.linalg.norm(points, axis=1)[:, None]
    truth = np.array([12, -2, 1.5])
    s = -U @ truth
    s[:450] += rng.normal(10, 3, 450)
    scan = Scan(points, s)
    velocity = estimate_ego_velocity(scan, Config())
    np.testing.assert_allclose(velocity, truth, atol=0.03)
    mask = dynamic_mask(scan, velocity, Config())
    assert mask[:450].mean() > 0.99
    assert not mask[450:].any()
    scan = Scan([[1, 0, 0], [100, 0, 0]], [0.6, 0.6])
    np.testing.assert_array_equal(dynamic_mask(scan, np.zeros(3), Config()), [True, False])


def test_full_3d_cluster_velocity_and_degeneracy():
    rng = np.random.default_rng(8)
    points = rng.normal(size=(150, 3)) + [8, 3, 2]
    U = points / np.linalg.norm(points, axis=1)[:, None]
    ego, obj = np.array([10, 1, 0.5]), np.array([3, -2, 4])
    scan = Scan(points, U @ (obj - ego))
    velocities, accepted = reconstruct_velocities(scan, np.zeros(150), ego, Config())
    np.testing.assert_allclose(velocities[0], obj, atol=1e-10)
    assert accepted.all()
    parallel = Scan(np.tile([20, 0, 0], (40, 1)), np.ones(40))
    velocities, accepted = reconstruct_velocities(parallel, np.zeros(40), np.zeros(3), Config())
    assert not velocities and not accepted.any()


def test_velocity_inlier_fraction_rejects_inconsistent_cluster():
    rng = np.random.default_rng(23)
    points = rng.normal(size=(100, 3)) + [8, 0, 0]
    scan = Scan(points, rng.normal(0, 20, 100))
    velocities, keep = reconstruct_velocities(scan, np.zeros(100), np.zeros(3), Config())
    assert not velocities and not keep.any()


def test_prediction_preserves_measurement_los_and_doppler(monkeypatch):
    rng = np.random.default_rng(10)
    points = rng.normal(size=(80, 3)) + [10, 0, 0]
    obj = np.array([8, 0, 3])
    U = points / np.linalg.norm(points, axis=1)[:, None]
    scan = Scan(points, U @ obj)
    monkeypatch.setattr("dynamic_icp.core.cluster_dynamics", lambda p, c: np.zeros(len(p), dtype=int))
    predicted, stats = predict_source(scan, np.zeros(3), 0.1, Config())
    np.testing.assert_allclose(predicted.points, points + obj * 0.1)
    np.testing.assert_allclose(predicted.los, scan.los, atol=1e-15)
    np.testing.assert_array_equal(predicted.doppler, scan.doppler)
    np.testing.assert_array_equal(scan.points, points)
    assert stats["predicted_dynamic"] == 80
    no_dpp, stats = predict_source(scan, np.zeros(3), 0.1, Config(ablation="no_dpp"))
    assert len(no_dpp.points) == 0 and stats["dynamic_candidates"] == 80
    no_vf, stats = predict_source(scan, np.zeros(3), 0.1, Config(ablation="no_vf"))
    np.testing.assert_array_equal(no_vf.points, points)
    assert stats["dynamic_candidates"] == 0


def test_voxel_keeps_los_doppler_association():
    scan = Scan([[1, 1, 1], [1.01, 1.01, 1.01], [3, 2, 1]], [2, 20, 5])
    down = scan.downsample(0.3)
    np.testing.assert_array_equal(down.doppler, [2, 5])
    np.testing.assert_allclose(down.los, scan.los[[0, 2]])


def test_real_hdbscan_and_prediction_of_two_moving_objects():
    rng = np.random.default_rng(46)
    p1 = rng.normal(0, 0.4, (80, 3)) + [10, 0, 0]
    p2 = rng.normal(0, 0.4, (80, 3)) + [10, 6, 0]
    points = np.vstack([p1, p2])
    objects = np.repeat([[6, 1, 2], [8, -1, 3]], 80, axis=0)
    U = points / np.linalg.norm(points, axis=1)[:, None]
    scan = Scan(points, np.sum(U * objects, axis=1))
    labels = cluster_dynamics(points, Config())
    assert len(np.unique(labels[labels >= 0])) == 2
    predicted, stats = predict_source(scan, np.zeros(3), 0.1, Config())
    assert stats["clusters"] == 2 and stats["predicted_dynamic"] == 160
    np.testing.assert_allclose(predicted.points, points + objects * 0.1, atol=1e-10)


def three_planes():
    rng = np.random.default_rng(12)
    points = []
    for axis in range(3):
        p = rng.uniform(-3, 3, size=(600, 3))
        p[:, axis] = 5
        points.append(p)
    return np.vstack(points)


def test_end_to_end_known_se3_and_failure_status():
    points = three_planes()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.015]).as_matrix()
    T[:3, 3] = [0.05, -0.06, 0.04]
    source = Scan(points, np.zeros(len(points)))
    target = Scan(points @ T[:3, :3].T + T[:3, 3], source.doppler)
    config = Config(voxel_size=0, normal_radius=0.5)
    result = register(source, target, 0.1, config)
    assert result.converged and result.num_iterations < config.max_iterations
    np.testing.assert_allclose(result.transformation, T, atol=2e-4)
    # Wrong direction cannot pass this asymmetric translation/rotation test.
    world_pose = np.linalg.inv(result.transformation)
    np.testing.assert_allclose(world_pose @ result.transformation, np.eye(4), atol=1e-12)
    bad_init = np.eye(4)
    bad_init[:3, 3] = 1000
    failure = register(source, target, 0.1, config, init_transform=bad_init)
    assert not failure.converged and failure.status == "insufficient_correspondences"
    assert failure.num_iterations == 0
    limited = register(source, target, 0.1, replace(config, max_iterations=1))
    assert not limited.converged and limited.status == "max_iterations"


def test_plane_degeneracy_is_not_reported_as_convergence():
    points = three_planes()[:600]
    scan = Scan(points, np.zeros(len(points)))
    result = register(scan, scan, 0.1, Config(voxel_size=0, normal_radius=0.5))
    assert not result.converged and result.status == "degenerate_system"


def test_tukey_outliers_and_invalid_input():
    np.testing.assert_allclose(tukey_weights(np.array([0, 0.3, -0.5, 2]), 0.3), [1, 0, 0, 0])
    with pytest.raises(ValueError):
        Scan([[0, 0, 0]], [1])
    with pytest.raises(ValueError):
        Config(lambda_doppler=1)
    with pytest.raises(ValueError):
        Config(min_samples=1.5)
    with pytest.raises(ValueError):
        Config(kappa=float("nan"))
