"""One dataset-independent runner for Dynamic-ICP and its paper ablations."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from .core import Config, register
from .io import evaluate_trajectory, load_scan, load_timestamps, save_tum, scan_paths


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scans", required=True, type=Path, help="Directory containing scans")
    ap.add_argument("--layout", choices=("xyzd", "aeva25", "aeva29", "npz"), default="xyzd")
    timing = ap.add_mutually_exclusive_group(required=True)
    timing.add_argument("--timestamps", type=Path, help="One timestamp in seconds per sorted scan")
    timing.add_argument("--filename-time-unit", choices=("s", "ms", "us", "ns"))
    timing.add_argument("--period", type=float, help="Constant frame period, seconds; time starts at zero")
    ap.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    ap.add_argument("--config", type=Path, help="JSON object overriding Config fields")
    ap.add_argument("--backend", choices=("numpy", "cpp"), help="Override the configured backend")
    ap.add_argument("--ablation", choices=("full", "no_vf", "no_dpp", "no_dr"))
    ap.add_argument("--start", type=int, default=0, help="First sorted scan (inclusive)")
    ap.add_argument("--stop", type=int, help="Last sorted scan (exclusive)")
    ap.add_argument("--doppler-sign", type=int, choices=(-1, 1), default=1,
                    help="Convert stored Doppler to positive-away convention")
    ap.add_argument("--gt", type=Path, help="Optional TUM GT, used only after registration")
    ap.add_argument("--gt-max-gap", type=float, default=0.2, help="Maximum interpolated GT gap, seconds")
    ap.add_argument("--gt-sensor-extrinsic", type=Path,
                    help="Optional JSON 4x4 T_body_sensor, only for evaluation of body-frame GT")
    return ap


def run(args):
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if args.ablation is not None:
        overrides["ablation"] = args.ablation
    if args.backend is not None:
        overrides["backend"] = args.backend
    config = Config(**overrides)
    paths = scan_paths(args.scans, args.layout)
    times = load_timestamps(paths, args.timestamps, args.filename_time_unit, args.period)
    stop = len(paths) if args.stop is None else args.stop
    if not (0 <= args.start < stop <= len(paths)) or stop - args.start < 2:
        raise ValueError("Selected range must contain at least two scans and be within the sequence")
    paths, times = paths[args.start:stop], times[args.start:stop]
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": asdict(config), "arguments": {k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()}, "scans": [str(p.resolve()) for p in paths],
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "hdbscan": version("hdbscan"),
                "backend": config.backend,
                "convergence_criterion": "absolute changes in fitness and inlier RMSE < tolerance"}
    if config.backend == "cpp":
        from .native_backend import load_backend
        _native = load_backend()
        metadata["native_solver_sha256"] = _native.solver_sha256
        metadata["native_module"] = _native.__file__
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    poses, rows = [np.eye(4)], []
    previous_velocity = None
    previous_rotation, previous_dt = None, None
    source = load_scan(paths[0], args.layout, args.doppler_sign)
    start_time = perf_counter()
    with (args.output / "pairs.jsonl").open("w") as log:
        for i in range(len(paths) - 1):
            target = load_scan(paths[i + 1], args.layout, args.doppler_sign)
            dt = float(times[i + 1] - times[i])
            before = perf_counter()
            rotation_prior = (None if previous_rotation is None else
                              Rotation.from_rotvec(previous_rotation * dt / previous_dt).as_matrix())
            result = register(source, target, dt, config, ego_prior=previous_velocity,
                              rotation_prior=rotation_prior)
            seconds = perf_counter() - before
            row = {"source": paths[i].name, "target": paths[i + 1].name,
                   "dt": dt, "seconds": seconds, "iterations": result.num_iterations,
                   "converged": result.converged, "status": result.status,
                   "fitness": result.fitness,
                   "inlier_rmse": result.inlier_rmse if np.isfinite(result.inlier_rmse) else None,
                   "ego_velocity": result.ego_velocity.tolist(),
                   "T_target_source": result.transformation.tolist(), **result.statistics}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if result.status not in ("converged", "max_iterations"):
                raise RuntimeError(f"Pair {i} failed: {result.status}; see {args.output / 'pairs.jsonl'}")
            # T maps old-frame points to new frame, so world poses compose its inverse.
            poses.append(poses[-1] @ np.linalg.inv(result.transformation))
            # Constant-velocity prior expressed in the new sensor coordinates.
            previous_velocity = -result.transformation[:3, 3] / dt
            previous_rotation = Rotation.from_matrix(result.transformation[:3, :3]).as_rotvec()
            previous_dt = dt
            source = target
            rows.append(row)
            print(f"[{i + 1}/{len(paths) - 1}] {result.status}; "
                  f"iterations={result.num_iterations}, fitness={result.fitness:.3f}", flush=True)
    elapsed = perf_counter() - start_time
    save_tum(args.output / "trajectory.tum", times, poses)
    summary = {"pairs": len(rows), "convergence_rate": float(np.mean([r['converged'] for r in rows])),
               "mean_iterations": float(np.mean([r['iterations'] for r in rows])),
               "registration_fps": len(rows) / sum(r["seconds"] for r in rows),
               "pipeline_fps": len(rows) / elapsed,
               "dynamic_fraction_mean": float(np.mean([r['dynamic_candidates'] / r['points'] for r in rows])),
               "dynamic_fraction_max": max(r['dynamic_candidates'] / r['points'] for r in rows)}
    if args.gt:
        extrinsic = json.loads(args.gt_sensor_extrinsic.read_text()) if args.gt_sensor_extrinsic else None
        summary["evaluation"] = evaluate_trajectory(times, poses, args.gt, args.gt_max_gap, extrinsic)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
