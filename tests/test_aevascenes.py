import numpy as np
from scipy.spatial.transform import Rotation

from dynamic_icp.aevascenes import load_compensated_frame


def test_compensated_loader_preserves_sensor_origin_and_doppler(tmp_path):
    points = np.array([[2.0, 1.0, -0.2], [3.0, -1.0, 0.4]])
    speed = np.array([4.0, -2.0])
    np.savez(
        tmp_path / "scan.npz",
        xyz=points,
        velocity=speed[:, None],
        semantic_labels=np.array(["car", "road"], dtype=object),
    )
    rotation = Rotation.from_euler("z", 80, degrees=True)
    translation = np.array([1.4, -0.3, 2.0])
    pose = {
        "rotation": dict(zip(("x", "y", "z", "w"), rotation.as_quat())),
        "translation": dict(zip(("x", "y", "z"), translation)),
    }
    frame = {"point_cloud": {"sensor": {"point_cloud_path": "scan.npz"}}}

    scan = load_compensated_frame(
        tmp_path,
        frame,
        {"vehicle_to_lidar_extrinsics": {"sensor": pose}},
        ["sensor"],
    )

    np.testing.assert_allclose(scan.points, rotation.apply(points) + translation)
    np.testing.assert_allclose(
        scan.los,
        rotation.apply(points / np.linalg.norm(points, axis=1)[:, None]),
    )
    np.testing.assert_array_equal(scan.doppler, speed)
    assert not np.allclose(
        scan.los, scan.points / np.linalg.norm(scan.points, axis=1)[:, None]
    )
