import numpy as np

from dynamic_icp.core import Config, Scan, predict_source
from dynamic_icp.demo_viewer import analyze_frame


def test_viewer_stages_match_registration_prediction(monkeypatch):
    rng = np.random.default_rng(81)
    static = rng.uniform([-20, -15, -2], [25, 15, 3], size=(600, 3))
    moving_1 = rng.normal(0, 0.4, size=(80, 3)) + [10, 0, 0]
    moving_2 = rng.normal(0, 0.4, size=(80, 3)) + [10, 6, 0]
    moving = np.vstack((moving_1, moving_2))
    points = np.vstack((static, moving))
    los = points / np.linalg.norm(points, axis=1)[:, None]
    object_velocity = np.repeat([[6, 1, 2], [8, -1, 3]], 80, axis=0)
    doppler = np.r_[
        np.zeros(len(static)),
        np.sum(los[len(static):] * object_velocity, axis=1),
    ]
    scan = Scan(points, doppler)
    config = Config(voxel_size=0)
    monkeypatch.setattr(
        "dynamic_icp.demo_viewer.estimate_ego_velocity",
        lambda scan, config: np.zeros(3),
    )

    stage = analyze_frame(scan, 0.1, config)
    predicted, statistics = predict_source(scan, stage.ego_velocity, 0.1, config)

    assert stage.accepted.sum() > 0
    np.testing.assert_allclose(predicted.points, stage.predicted_points[stage.keep])
    np.testing.assert_array_equal(predicted.doppler, stage.scan.doppler[stage.keep])
    assert statistics["dynamic_candidates"] == int(stage.dynamic.sum())
    assert statistics["predicted_dynamic"] == int(stage.accepted.sum())
    assert statistics["clusters"] == len(stage.velocities)
