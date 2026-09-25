import json

import numpy as np

from dynamic_icp.cli import parser, run
from dynamic_icp.io import load_tum, save_tum
from test_core import three_planes


def test_cli_trajectory_direction_evaluation_and_no_overwrite(tmp_path):
    import pytest

    points = three_planes()
    folder = tmp_path / "scans"
    folder.mkdir()
    np.savez(folder / "0.npz", points=points, doppler=np.zeros(len(points)))
    shift = np.array([0.05, -0.02, 0.01])
    np.savez(folder / "1.npz", points=points - shift, doppler=np.zeros(len(points)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"voxel_size": 0, "normal_radius": 0.5}))
    gt = np.repeat(np.eye(4)[None], 2, axis=0)
    gt[1, :3, 3] = shift
    gt_path = tmp_path / "gt.tum"
    save_tum(gt_path, [0, 0.1], gt)
    args = parser().parse_args(["--scans", str(folder), "--layout", "npz", "--period", "0.1",
                               "--config", str(config), "--output", str(tmp_path / "out"),
                               "--gt", str(gt_path)])
    summary = run(args)
    assert summary["evaluation"]["rte_rmse_m"] < 1e-5
    assert summary["convergence_rate"] == 1
    _, poses = load_tum(args.output / "trajectory.tum")
    np.testing.assert_allclose(poses[1], gt[1], atol=1e-5)
    metadata = json.loads((args.output / "run.json").read_text())
    assert metadata["config"]["lambda_doppler"] == 0.2
    with pytest.raises(ValueError, match="not empty"):
        run(args)
