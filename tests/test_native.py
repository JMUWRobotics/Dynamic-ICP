"""Run after building native/; portable installations can skip these tests."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dynamic_icp import Config, Scan, register
from dynamic_icp.core import displacement_residual, residuals_and_jacobians, tukey_weights
from test_core import three_planes

native = pytest.importorskip("dynamic_icp._native")


@pytest.mark.parametrize("weight,ego_weight", [(0.0, 0.0), (0.2, 0.0), (0.8, 0.0), (0.2, 2.0)])
def test_compiled_solver_step_matches_equations_at_nonidentity_pose(weight, ego_weight):
    rng = np.random.default_rng(91)
    p, q, up, uq, normals = [rng.normal(size=(100, 3)) for _ in range(5)]
    for vectors in (up, uq, normals):
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    sp, sq = rng.normal(size=(2, 100))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.12, -0.07, 0.23]).as_matrix()
    T[:3, 3] = [0.4, -0.6, 0.2]
    rg, rv, Jg, Jv = residuals_and_jacobians(p, up, sp, q, normals, uq, sq, T)
    wg = np.sqrt((1 - weight) * tukey_weights(rg, 5))
    wv = np.sqrt(weight * tukey_weights(rv, 2))
    A = np.vstack([Jg * wg[:, None], Jv * wv[:, None]])
    b = -np.r_[rg * wg, rv * wv]
    displacement = np.array([0.3, -0.1, 0.2])
    information = np.diag([0.8, 0.15, 0.05])
    if ego_weight:
        r, J = displacement_residual(T, displacement)
        root = np.sqrt(ego_weight * len(p)) * np.sqrt(information)
        A = np.vstack((A, root @ J))
        b = np.r_[b, -root @ r]
    delta = np.linalg.lstsq(A, b, rcond=None)[0]
    expected = T.copy()
    expected[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ T[:3, :3]
    expected[:3, 3] += delta[3:]
    actual = native.step(p, sp, up, q, sq, uq, normals, T, weight, 5, 2,
                         ego_weight, displacement, information) @ T
    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_compiled_registration_and_iteration_failure_statuses():
    p = three_planes()
    source = Scan(p, np.zeros(len(p)))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.015]).as_matrix()
    T[:3, 3] = [0.05, -0.06, 0.04]
    target = Scan(p @ T[:3, :3].T + T[:3, 3], source.doppler)
    cfg = Config(backend="cpp", voxel_size=0, normal_radius=0.5)
    cpp = register(source, target, 0.1, cfg)
    python = register(source, target, 0.1, replace(cfg, backend="numpy"))
    assert cpp.converged and cpp.num_iterations == python.num_iterations
    np.testing.assert_allclose(cpp.transformation, python.transformation, atol=1e-10)
    limited = register(source, target, 0.1, replace(cfg, max_iterations=1))
    assert not limited.converged and limited.num_iterations == 1
    plane = source.select(slice(0, 600))
    assert register(plane, plane, 0.1, cfg).status == "degenerate_system"
    init = np.eye(4)
    init[:3, 3] = 1000
    assert register(source, target, 0.1, cfg, init_transform=init).status == "insufficient_correspondences"
    # A configured low-speed gate suppresses the additional prior on a static scan.
    gated = register(source, target, 0.1, replace(cfg, ego_translation_weight=10, ego_prior_min_speed=0.5))
    np.testing.assert_allclose(gated.transformation, cpp.transformation, atol=1e-11)
