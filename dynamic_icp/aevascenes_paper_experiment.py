"""Run the paper-style full-3D Dynamic-ICP method on one AevaScenes sequence."""

import argparse
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .aevascenes import load_compensated_frame, pose_matrix
from .core import Config, dynamic_mask, estimate_normals, predict_source
from .io import evaluate_trajectory, load_tum, save_tum
from .native_backend import load_backend


def _solve(native, source, target, normals, init, config, gate=None, doppler_weight=None):
    """Run the native solver with the paper objective and no motion prior."""
    return native.register(
        source.points,
        source.doppler,
        source.los,
        target.points,
        target.doppler,
        target.los,
        normals,
        init,
        config.max_correspondence_distance if gate is None else gate,
        config.lambda_doppler if doppler_weight is None else doppler_weight,
        config.geometric_k,
        config.doppler_k,
        config.tolerance,
        config.max_iterations,
        0.0,
        np.zeros(3),
        np.zeros((3, 3)),
    )


def _static_initialization(native, source, target, initial, config):
    """Compute a coarse geometric seed from Doppler-classified static points."""
    source = source.select(~dynamic_mask(source, np.zeros(3), config)).downsample(1.0)
    target = target.select(~dynamic_mask(target, np.zeros(3), config)).downsample(1.0)
    coarse = replace(config, normal_radius=3.0, geometric_k=1.5, max_iterations=40)
    normals = estimate_normals(target.points, coarse)
    valid = np.linalg.norm(normals, axis=1) > 0
    target, normals = target.select(valid), normals[valid]
    transform = initial.copy()
    for gate in (3.0, 1.0):
        transform = _solve(
            native,
            source,
            target,
            normals,
            transform,
            coarse,
            gate=gate,
            doppler_weight=0.0,
        )["transformation"]
    return transform


def _evaluate_pairs(times, transforms, gt_path):
    """Evaluate every adjacent transform and return per-pair error details."""
    gt_times, gt = load_tum(gt_path)
    timestamps = np.asarray(times, dtype=float)
    right = np.searchsorted(gt_times, timestamps)
    left = np.clip(right - 1, 0, len(gt_times) - 1)
    right = np.clip(right, 0, len(gt_times) - 1)
    nearest = np.where(
        np.abs(timestamps - gt_times[left]) < np.abs(timestamps - gt_times[right]),
        left,
        right,
    )
    roundoff = max(1e-9, 4 * abs(np.spacing(np.max(np.abs(gt_times)))))
    timestamps = np.where(
        np.abs(timestamps - gt_times[nearest]) <= roundoff,
        gt_times[nearest],
        timestamps,
    )
    if timestamps[0] < gt_times[0] or timestamps[-1] > gt_times[-1]:
        raise ValueError("Evaluation interval extends outside ground truth")

    reference = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
    reference[:, :3, :3] = Slerp(
        gt_times, Rotation.from_matrix(gt[:, :3, :3])
    )(timestamps).as_matrix()
    for axis in range(3):
        reference[:, axis, 3] = np.interp(
            timestamps, gt_times, gt[:, axis, 3]
        )
    relative = np.linalg.inv(reference[:-1]) @ reference[1:]
    errors = np.linalg.inv(relative) @ np.linalg.inv(transforms)
    rte = np.linalg.norm(errors[:, :3, 3], axis=1)
    rotation_error = np.degrees(
        Rotation.from_matrix(errors[:, :3, :3]).as_rotvec()
    )
    rre = np.linalg.norm(rotation_error, axis=1)

    poses = [np.eye(4)]
    for transform in transforms:
        poses.append(poses[-1] @ np.linalg.inv(transform))
    summary = evaluate_trajectory(times, poses, gt_path)
    if summary["pairs"] != len(transforms):
        raise ValueError("Ground truth does not support every adjacent pair")
    np.testing.assert_allclose(
        [np.sqrt(np.mean(rte**2)), np.sqrt(np.mean(rre**2))],
        [summary["rte_rmse_m"], summary["rre_rmse_deg"]],
        rtol=1e-6,
        atol=1e-8,
    )
    details = {
        "rte_m": rte.tolist(),
        "rre_deg": rre.tolist(),
        "rotation_error_vector_deg": rotation_error.tolist(),
        "gt_rotation_deg": np.degrees(
            Rotation.from_matrix(relative[:, :3, :3]).magnitude()
        ).tolist(),
    }
    return summary, details, poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/datasets/reference.json")
    )
    parser.add_argument(
        "--sensors", nargs="+", default=("front_wide_lidar", "front_narrow_lidar")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.pairs < 1:
        parser.error("--start must be nonnegative and --pairs must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)

    sequence_path = args.sequence / "sequence.json"
    sequence = json.loads(sequence_path.read_text())
    frames = sorted(sequence["frames"], key=lambda frame: int(frame["timestamp_ns"]))
    frames = frames[args.start : args.start + args.pairs + 1]
    if len(frames) != args.pairs + 1:
        parser.error("Not enough frames for the requested interval")
    times = [Decimal(frame["timestamp_ns"]) / Decimal(10**9) for frame in frames]
    values = json.loads(args.config.read_text())
    values["backend"] = "cpp"
    config, native = Config(**values), load_backend()
    sources = (
        Path(__file__),
        Path(__file__).with_name("core.py"),
        Path(__file__).with_name("aevascenes.py"),
        Path(__file__).with_name("io.py"),
    )
    protocol = {
        "sequence": str(args.sequence),
        "sequence_metadata_sha256": hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
        "start": args.start,
        "pairs": args.pairs,
        # argparse preserves the tuple default but returns a list when the
        # option is supplied.  Normalize it so a JSON round trip compares
        # equal when a partial run is resumed.
        "sensors": list(args.sensors),
        "config": asdict(config),
        "method": "paper full-3D clustered velocity reconstruction and prediction",
        "doppler_convention": "released ego-motion-compensated Doppler; ego velocity set to zero",
        "ground_truth_usage": "evaluation only, after all registration",
        "native_solver_sha256": native.solver_sha256,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
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
    transforms = [np.asarray(row["transform"], dtype=float) for row in rows]
    previous = np.eye(4) if not rows else np.asarray(rows[-1]["static_pose"], dtype=float)
    source = load_compensated_frame(
        args.sequence, frames[len(rows)], sequence["metadata"], args.sensors
    ).downsample(config.voxel_size)

    for pair_index in range(len(rows), args.pairs):
        target = load_compensated_frame(
            args.sequence, frames[pair_index + 1], sequence["metadata"], args.sensors
        ).downsample(config.voxel_size)
        dt = float(times[pair_index + 1] - times[pair_index])
        coarse = _static_initialization(native, source, target, previous, config)
        normals = estimate_normals(target.points, config)
        valid = np.linalg.norm(normals, axis=1) > 0
        match_target, normals = target.select(valid), normals[valid]
        dynamic = dynamic_mask(source, np.zeros(3), config)
        static_pose = _solve(
            native,
            source.select(~dynamic),
            match_target,
            normals,
            coarse,
            config,
        )["transformation"]
        predicted, statistics = predict_source(source, np.zeros(3), dt, config)
        transform = _solve(
            native, predicted, match_target, normals, static_pose, config
        )["transformation"]
        transforms.append(transform)
        row = {
            "pair": args.start + pair_index,
            "dt": dt,
            "statistics": statistics,
            "static_pose": static_pose.tolist(),
            "transform": transform.tolist(),
        }
        with progress_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        previous = static_pose
        source = target
        if pair_index == 0 or (pair_index + 1) % 10 == 0 or pair_index + 1 == args.pairs:
            print(f"{pair_index + 1}/{args.pairs}", statistics, flush=True)

    save_tum(
        args.output / "gt.tum", times, [pose_matrix(frame["ego_pose"]) for frame in frames]
    )
    array = np.asarray(transforms)
    summary, details, poses = _evaluate_pairs(times, array, args.output / "gt.tum")
    results = {"paper_dynamic": summary}
    (args.output / "paper_dynamic.json").write_text(json.dumps(details, indent=2) + "\n")
    save_tum(args.output / "paper_dynamic.tum", times, poses)
    np.savez_compressed(args.output / "transforms.npz", paper_dynamic=array)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
