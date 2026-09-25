from decimal import Decimal

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dynamic_icp.io import evaluate_trajectory, load_scan, load_timestamps, save_tum, scan_paths


def test_binary_layouts_and_invalid_returns(tmp_path):
    path = tmp_path / "scan.bin"
    np.array([[1, 2, 3, -4], [0, 0, 0, 5], [np.nan, 0, 0, 1]], dtype="<f4").tofile(path)
    scan = load_scan(path)
    np.testing.assert_array_equal(scan.points, [[1, 2, 3]])
    np.testing.assert_array_equal(scan.doppler, [-4])
    np.testing.assert_array_equal(load_scan(path, doppler_sign=-1).doppler, [4])
    for size in (25, 29):
        record = np.zeros(1, dtype=np.dtype({"names": ["x", "y", "z", "v"],
                             "formats": ["<f4"] * 4, "offsets": [0, 4, 8, 16], "itemsize": size}))
        record["x"], record["v"] = 10, -7
        record.tofile(path)
        native = load_scan(path, f"aeva{size}")
        np.testing.assert_array_equal(native.points, [[10, 0, 0]])
        np.testing.assert_array_equal(native.doppler, [-7])
    with pytest.raises(ValueError):
        load_scan(path, "xyzd")


def test_npz_separate_los(tmp_path):
    path = tmp_path / "scan.npz"
    np.savez(path, points=[[10, 1, 1]], doppler=[3], los=[[1, 0, 0]])
    scan = load_scan(path, "npz")
    np.testing.assert_array_equal(scan.los, [[1, 0, 0]])


def test_numeric_scan_order_nanosecond_precision_and_count(tmp_path):
    for name in ("10", "2", "1"):
        (tmp_path / f"{name}.bin").touch()
    assert [p.stem for p in scan_paths(tmp_path, "xyzd")] == ["1", "2", "10"]
    paths = [tmp_path / f"{n}.bin" for n in (1724659438133859556, 1724659438236240483)]
    stamps = load_timestamps(paths, filename_unit="ns")
    assert stamps[1] - stamps[0] == Decimal("0.102380927")
    f = tmp_path / "times.txt"
    f.write_text("0\n")
    with pytest.raises(ValueError, match="count"):
        load_timestamps(paths, timestamp_file=f)
    with pytest.raises(ValueError, match="increasing"):
        load_timestamps(paths, period=0)


def test_rpe_translation_rotation_interpolation_and_gap(tmp_path):
    times = [0, 1, 2]
    gt = np.repeat(np.eye(4)[None], 3, axis=0)
    gt[:, 0, 3] = times
    path = tmp_path / "gt.tum"
    save_tum(path, times, gt)
    estimate = gt.copy()
    estimate[:, 0, 3] *= 1.1
    result = evaluate_trajectory(times, estimate, path)
    assert result["rte_rmse_m"] == pytest.approx(0.1)
    assert result["rre_rmse_deg"] == pytest.approx(0)
    # Estimate at an unsupported timestamp cannot create a longer frame-gap pair.
    with pytest.raises(ValueError, match="No adjacent"):
        evaluate_trajectory([0, 0.5, 2], estimate, path, max_gap=0.2)
    interpolated = gt.copy()
    interpolated[:, 0, 3] = [0, 0.5, 2]
    result = evaluate_trajectory([0, 0.5, 2], interpolated, path, max_gap=1.1)
    assert result["rte_mean_m"] == pytest.approx(0)
    estimate = gt.copy()
    estimate[1:, :3, :3] = Rotation.from_euler("z", [1, 2], degrees=True).as_matrix()
    result = evaluate_trajectory(times, estimate, path)
    assert result["rre_mean_deg"] == pytest.approx(1)


def test_epoch_roundoff_does_not_drop_first_pose(tmp_path):
    times = np.array([1724479473.874276, 1724479473.974276])
    gt = np.repeat(np.eye(4)[None], 2, axis=0)
    path = tmp_path / "gt.tum"
    save_tum(path, times, gt)
    times[0] -= np.spacing(times[0])
    result = evaluate_trajectory(times, gt, path)
    assert result["pairs"] == 1 and result["matched_poses"] == 2
