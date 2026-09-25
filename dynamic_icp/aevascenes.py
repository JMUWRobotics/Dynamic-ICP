"""Read released AevaScenes v0.1 compensated scans without using pose/box labels.

The SDK applies metadata['vehicle_to_lidar_extrinsics'] to LiDAR points to
obtain vehicle coordinates, despite the field's potentially confusing name.
The released compensated XYZ only permits approximate measurement LOS; retain
each sensor's origin instead of normalizing merged vehicle-frame positions.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .core import Scan


def pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([pose["rotation"][k] for k in ("x", "y", "z", "w")]).as_matrix()
    T[:3, 3] = [pose["translation"][k] for k in ("x", "y", "z")]
    return T


def load_compensated_frame(directory, frame, metadata, sensors):
    points, velocities, directions = [], [], []
    for sensor in sensors:
        path = Path(directory) / frame["point_cloud"][sensor]["point_cloud_path"]
        with np.load(path, allow_pickle=False) as data:
            xyz = np.asarray(data["xyz"], dtype=float)
            velocity = np.asarray(data["velocity"], dtype=float).reshape(-1)
        if xyz.shape != (len(velocity), 3):
            raise ValueError(f"Invalid point/velocity shapes: {path}")
        ranges = np.linalg.norm(xyz, axis=1)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(velocity) & (ranges >= .2) & (ranges <= 400)
        xyz, velocity, ranges = xyz[valid], velocity[valid], ranges[valid]
        T = pose_matrix(metadata["vehicle_to_lidar_extrinsics"][sensor])
        points.append(xyz @ T[:3, :3].T + T[:3, 3])
        directions.append((xyz / ranges[:, None]) @ T[:3, :3].T)
        velocities.append(velocity)
    return Scan(np.concatenate(points), np.concatenate(velocities), np.concatenate(directions))
