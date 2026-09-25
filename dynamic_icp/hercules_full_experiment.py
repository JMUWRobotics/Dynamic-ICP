"""Run the paper Dynamic-ICP method over one full HeRCULES sequence."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .core import Config, register
from .io import evaluate_trajectory, load_scan, load_timestamps, save_tum, scan_paths
from .native_backend import load_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/datasets/reference.json")
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.pairs < 1:
        parser.error("--start must be nonnegative and --pairs must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)

    paths = scan_paths(args.scans, "aeva29")
    all_times = load_timestamps(paths, filename_unit="ns")
    stop = args.start + args.pairs + 1
    if stop > len(paths):
        parser.error("Not enough scans for the requested interval")
    paths, times = paths[args.start:stop], all_times[args.start:stop]

    values = json.loads(args.config.read_text())
    values["backend"] = "cpp"
    configs = {"paper_dynamic": Config(**values)}
    native = load_backend()
    protocol = {
        "scans": str(args.scans.resolve()),
        "gt": str(args.gt.resolve()),
        "layout": "aeva29",
        "filename_time_unit": "ns",
        "start": args.start,
        "pairs": args.pairs,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "ground_truth_usage": "evaluation only, after all registration",
        "native_solver_sha256": native.solver_sha256,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("core.py"))
        },
    }
    protocol_path = args.output / "protocol.json"
    progress_path = args.output / "progress.jsonl"
    rows = []
    if args.resume and protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            parser.error("Existing output protocol differs from the requested run")
        if (args.output / "results.json").exists():
            print((args.output / "results.json").read_text(), end="")
            return
        if progress_path.exists():
            rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    if len(rows) > args.pairs:
        parser.error("Checkpoint contains more pairs than requested")

    transforms = {name: [] for name in configs}
    for row in rows:
        for name in configs:
            transforms[name].append(np.asarray(row["transforms"][name], dtype=float))
    priors = {name: None for name in configs}
    rotations = {name: None for name in configs}
    previous_dt = None
    if rows:
        previous_dt = float(rows[-1]["dt"])
        for name in configs:
            transform = transforms[name][-1]
            priors[name] = -transform[:3, 3] / previous_dt
            rotations[name] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()

    source = load_scan(paths[len(rows)], "aeva29")
    for pair_index in range(len(rows), args.pairs):
        target = load_scan(paths[pair_index + 1], "aeva29")
        dt = float(times[pair_index + 1] - times[pair_index])
        row = {"pair": args.start + pair_index, "dt": dt, "variants": {}, "transforms": {}}
        for name, config in configs.items():
            rotation_prior = (
                None
                if rotations[name] is None
                else Rotation.from_rotvec(rotations[name] * dt / previous_dt).as_matrix()
            )
            result = register(
                source,
                target,
                dt,
                config,
                ego_prior=priors[name],
                rotation_prior=rotation_prior,
            )
            if result.status not in ("converged", "max_iterations"):
                raise RuntimeError(f"{name} pair {args.start + pair_index}: {result.status}")
            transform = result.transformation
            transforms[name].append(transform)
            row["transforms"][name] = transform.tolist()
            row["variants"][name] = {
                "status": result.status,
                "iterations": result.num_iterations,
                "converged": result.converged,
                "fitness": result.fitness,
                **result.statistics,
            }
            priors[name] = -transform[:3, 3] / dt
            rotations[name] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
        with progress_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        previous_dt = dt
        source = target
        if pair_index == 0 or (pair_index + 1) % 50 == 0 or pair_index + 1 == args.pairs:
            print(f"{pair_index + 1}/{args.pairs}", flush=True)

    summaries = {}
    for name, values in transforms.items():
        poses = [np.eye(4)]
        for transform in values:
            poses.append(poses[-1] @ np.linalg.inv(transform))
        save_tum(args.output / f"{name}.tum", times, poses)
        summaries[name] = evaluate_trajectory(times, poses, args.gt)
    np.savez_compressed(
        args.output / "transforms.npz",
        **{name: np.asarray(values) for name, values in transforms.items()},
    )
    (args.output / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
