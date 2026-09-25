"""Explicit scan layouts and time units; no dependency on custom Open3D fields."""

from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .core import Scan


def load_scan(path, layout="xyzd", doppler_sign=1.0, min_range=0.2, max_range=400.0):
    path = Path(path)
    offsets = None
    if layout == "npz":
        with np.load(path, allow_pickle=False) as data:
            points, doppler = data["points"], data["doppler"]
            los = data["los"] if "los" in data else None
            offsets = data["time_offsets"] if "time_offsets" in data else None
    elif layout == "xyzd":
        if path.stat().st_size % 16:
            raise ValueError(f"{path}: expected 16-byte float32 x,y,z,Doppler records")
        data = np.fromfile(path, dtype="<f4").reshape(-1, 4)
        points, doppler, los = data[:, :3], data[:, 3], None
    elif layout in ("aeva25", "aeva29"):
        # Packed layouts used by the repository's HeRCULES/HeLiPR converters.
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                  ("reflectivity", "<f4"), ("velocity", "<f4"),
                  ("time_offset_ns", "<i4"), ("line_index", "u1")]
        if layout == "aeva29":
            fields.append(("intensity", "<f4"))
        dtype = np.dtype(fields)
        if path.stat().st_size % dtype.itemsize:
            raise ValueError(f"{path}: not a multiple of {dtype.itemsize} bytes")
        data = np.fromfile(path, dtype=dtype)
        points = np.column_stack([data[k] for k in ("x", "y", "z")])
        doppler, los = data["velocity"], None
        offsets = data["time_offset_ns"].astype(float) * 1e-9
    else:
        raise ValueError(f"Unknown scan layout: {layout}")
    points, doppler = np.asarray(points, dtype=float), np.asarray(doppler, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape != (len(doppler), 3):
        raise ValueError("Invalid point/Doppler shape")
    ranges = np.linalg.norm(points, axis=1)
    keep = (np.isfinite(points).all(axis=1) & np.isfinite(doppler)
            & (ranges >= min_range) & (ranges <= max_range) & (ranges > 0))
    if los is not None:
        los = np.asarray(los, dtype=float)
        if los.shape != points.shape:
            raise ValueError("LOS shape does not match points")
        keep &= np.isfinite(los).all(axis=1) & (np.linalg.norm(los, axis=1) > 0)
        los = los[keep]
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (len(points),):
            raise ValueError("Acquisition offset count does not match points")
        offsets = offsets[keep]
    return Scan(points[keep], doppler[keep] * doppler_sign, los, offsets)


def scan_paths(directory, layout):
    paths = list(Path(directory).glob("*.npz" if layout == "npz" else "*.bin"))
    if not paths:
        raise ValueError(f"No scans found in {directory}")
    try:
        return sorted(paths, key=lambda p: Decimal(p.stem))
    except Exception:
        return sorted(paths)


def load_timestamps(paths, timestamp_file=None, filename_unit=None, period=None):
    """Return decimal seconds to avoid losing precision in nanosecond filenames."""
    if sum(x is not None for x in (timestamp_file, filename_unit, period)) != 1:
        raise ValueError("Choose exactly one timestamp source")
    if timestamp_file is not None:
        # One timestamp per sorted scan. A TUM file's first column is accepted.
        lines = Path(timestamp_file).read_text().splitlines()
        times = [Decimal(line.split()[0]) for line in lines
                 if line.strip() and not line.lstrip().startswith("#")]
    elif filename_unit is not None:
        scale = {"s": Decimal(1), "ms": Decimal("0.001"),
                 "us": Decimal("0.000001"), "ns": Decimal("0.000000001")}[filename_unit]
        times = [Decimal(p.stem) * scale for p in paths]
    else:
        if not np.isfinite(period) or period <= 0:
            raise ValueError("Period must be positive and finite for increasing timestamps")
        times = [i * Decimal(str(period)) for i in range(len(paths))]
    if len(times) != len(paths):
        raise ValueError("Timestamp count must equal scan count; do not index unrelated GT rows")
    if not all(t.is_finite() for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Timestamps must be finite and strictly increasing")
    return times


def save_tum(path, timestamps, poses):
    if len(timestamps) != len(poses):
        raise ValueError("Timestamp and pose counts must match")
    with Path(path).open("w") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw; T_world_sensor\n")
        for timestamp, pose in zip(timestamps, poses):
            values = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
            stream.write(f"{timestamp:.9f} " + " ".join(f"{x:.12g}" for x in values) + "\n")


def load_tum(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError("TUM data must have eight finite columns in seconds")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("TUM timestamps must be strictly increasing")
    poses = np.repeat(np.eye(4)[None], len(data), axis=0)
    poses[:, :3, 3] = data[:, 1:4]
    poses[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
    return data[:, 0], poses


def evaluate_trajectory(timestamps, poses, gt_path, max_gap=0.2, gt_sensor_extrinsic=None):
    """Adjacent estimated-frame RPE; interpolate GT without crossing large gaps.

    GT must be T_world_sensor, or T_world_body with T_body_sensor supplied.
    Unmatched estimates break the evaluation chain (never bridge a dropped frame).
    """
    gt_times, gt = load_tum(gt_path)
    if len(gt_times) < 2 or max_gap <= 0:
        raise ValueError("Need at least two GT poses and a positive max_gap")
    times = np.asarray(timestamps, dtype=float)
    if len(times) != len(poses) or np.any(np.diff(times) <= 0):
        raise ValueError("Estimated timestamps must match poses and be increasing")
    right = np.searchsorted(gt_times, times)
    left_index = np.clip(right - 1, 0, len(gt_times) - 1)
    right_index = np.clip(right, 0, len(gt_times) - 1)
    nearest = np.where(np.abs(times - gt_times[left_index]) < np.abs(times - gt_times[right_index]),
                       left_index, right_index)
    # Epoch seconds have ~0.24 us float spacing. Do not lose an endpoint to
    # decimal serialization roundoff, or confuse this with a real data gap.
    roundoff = max(1e-9, 4 * abs(np.spacing(np.max(np.abs(gt_times)))))
    exact = np.abs(times - gt_times[nearest]) <= roundoff
    times = np.where(exact, gt_times[nearest], times)
    right = np.clip(right, 1, len(gt_times) - 1)
    valid = ((times >= gt_times[0]) & (times <= gt_times[-1])
             & (exact | ((gt_times[right] - gt_times[right - 1]) <= max_gap)))
    sample_times = np.clip(times, gt_times[0], gt_times[-1])
    reference = np.repeat(np.eye(4)[None], len(times), axis=0)
    for axis in range(3):
        reference[:, axis, 3] = np.interp(sample_times, gt_times, gt[:, axis, 3])
    reference[:, :3, :3] = Slerp(gt_times, Rotation.from_matrix(gt[:, :3, :3]))(sample_times).as_matrix()
    if gt_sensor_extrinsic is not None:
        extrinsic = np.asarray(gt_sensor_extrinsic, dtype=float)
        if (extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all()
                or not np.allclose(extrinsic[3], [0, 0, 0, 1])
                or not np.allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(extrinsic[:3, :3]), 1)):
            raise ValueError("GT sensor extrinsic must be a valid 4x4 T_body_sensor")
        reference = reference @ extrinsic
    rows = []
    for i in range(len(times) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        ref_relative = np.linalg.inv(reference[i]) @ reference[i + 1]
        est_relative = np.linalg.inv(poses[i]) @ poses[i + 1]
        error = np.linalg.inv(ref_relative) @ est_relative
        rows.append((float(np.linalg.norm(error[:3, 3])),
                     float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))))
    if not rows:
        raise ValueError("No adjacent estimated frames have supported GT timestamps")
    errors = np.asarray(rows)
    return {"pairs": len(rows), "matched_poses": int(valid.sum()),
            "rte_mean_m": float(errors[:, 0].mean()),
            "rte_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "rre_mean_deg": float(errors[:, 1].mean()),
            "rre_rmse_deg": float(np.sqrt(np.mean(errors[:, 1] ** 2)))}
