"""Equations (3)--(14) of docs/dynamic_icp_paper.pdf.

Points, LOS and velocities live in the sensor frame. T maps source to target.
The optimizer uses left SO(3) increments and additive target-frame translation;
the Doppler Jacobian therefore has three exactly zero translation columns.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Config:
    backend: str = "numpy"
    use_rotation_prior: bool = False
    ego_translation_weight: float = 0.0
    ego_prior_min_speed: float = 0.0
    deskew: bool = False
    # Values explicitly specified in Sec. IV-A and Table VI.
    lambda_doppler: float = 0.2
    geometric_k: float = 0.5
    doppler_k: float = 0.3
    min_cluster_size: int = 30
    min_samples: int = 10
    max_iterations: int = 100
    tolerance: float = 1e-5
    max_correspondence_distance: float = 0.3
    # Engineering defaults: the paper does not give numeric values for these.
    voxel_size: float = 0.3
    normal_radius: float = 2.0
    normal_max_nn: int = 30
    huber_delta: float = 0.3
    ego_iterations: int = 50
    tau0: float = 0.35
    kappa: float = 0.005
    velocity_relative_threshold: float = 0.05
    min_inlier_fraction: float = 0.5
    min_singular_value: float = 1e-3
    max_condition_number: float = 400.0
    # Optional distance-aware gate; zero keeps the paper's 0.3 m benchmark gate.
    correspondence_range_scale: float = 0.0
    ablation: str = "full"

    def __post_init__(self):
        if self.backend not in ("numpy", "cpp"):
            raise ValueError("backend must be numpy or cpp")
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.lambda_doppler < 1:
            raise ValueError("lambda_doppler must be in [0, 1)")
        for name in ("geometric_k", "doppler_k", "tolerance", "normal_radius",
                     "huber_delta", "max_correspondence_distance", "tau0",
                     "velocity_relative_threshold", "min_singular_value"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, minimum in (("min_cluster_size", 2), ("min_samples", 1),
                              ("max_iterations", 1), ("ego_iterations", 1),
                              ("normal_max_nn", 3)):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if min(self.voxel_size, self.kappa, self.correspondence_range_scale,
               self.ego_translation_weight, self.ego_prior_min_speed) < 0:
            raise ValueError("voxel_size and range scales must be nonnegative")
        if not 0 < self.min_inlier_fraction <= 1 or self.max_condition_number < 1:
            raise ValueError("Invalid cluster validation thresholds")
        if self.ablation not in ("full", "no_vf", "no_dpp", "no_dr"):
            raise ValueError("Unknown ablation")


@dataclass
class Scan:
    points: np.ndarray
    doppler: np.ndarray
    los: Optional[np.ndarray] = None
    time_offsets: Optional[np.ndarray] = None

    def __post_init__(self):
        self.points = np.array(self.points, dtype=float, copy=True)
        self.doppler = np.array(self.doppler, dtype=float, copy=True).reshape(-1)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if len(self.points) != len(self.doppler):
            raise ValueError("Each point must have one Doppler measurement")
        if not np.isfinite(self.points).all() or not np.isfinite(self.doppler).all():
            raise ValueError("Scan contains nonfinite values")
        if self.los is None:
            self.los = self.points.copy()
        else:
            self.los = np.array(self.los, dtype=float, copy=True)
        if self.los.shape != self.points.shape or not np.isfinite(self.los).all():
            raise ValueError("LOS must be finite and have shape (N, 3)")
        lengths = np.linalg.norm(self.los, axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError("Zero-length LOS is undefined")
        self.los /= lengths[:, None]
        if self.time_offsets is not None:
            self.time_offsets = np.asarray(self.time_offsets, dtype=float).copy()
            if self.time_offsets.shape != (len(self.points),) or not np.isfinite(self.time_offsets).all():
                raise ValueError("time_offsets must be one finite acquisition offset in seconds per point")

    def select(self, indices):
        return Scan(self.points[indices], self.doppler[indices], self.los[indices],
                    None if self.time_offsets is None else self.time_offsets[indices])

    def downsample(self, voxel_size):
        """One original return per voxel: preserve measured Doppler and LOS."""
        if voxel_size <= 0 or not len(self.points):
            return self.select(slice(None))
        _, indices = np.unique(np.floor(self.points / voxel_size), axis=0, return_index=True)
        return self.select(np.sort(indices))


def estimate_ego_velocity(scan, config, prior=None):
    """Huber IRLS for s + Uv (Eq. 3), initialized from the previous pose."""
    if len(scan.points) < 3:
        raise ValueError("At least three points are needed to estimate ego velocity")
    velocity = np.zeros(3) if prior is None else np.array(prior, dtype=float, copy=True)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("Ego prior must be a finite 3-vector")
    for _ in range(config.ego_iterations):
        residual = scan.doppler + scan.los @ velocity
        weights = np.minimum(1.0, config.huber_delta / np.maximum(np.abs(residual), 1e-12))
        A = scan.los * np.sqrt(weights[:, None])
        # Incremental solve preserves the prior in unobservable directions.
        delta = np.linalg.lstsq(A, -residual * np.sqrt(weights), rcond=None)[0]
        velocity += delta
        if np.linalg.norm(delta) < config.tolerance:
            break
    return velocity


def deskew_scan(scan, velocity, angular_velocity):
    """Move measured returns/LOS to scan-start coordinates, without using GT.

    This optional constant-twist approximation is additional to the paper.
    Doppler remains the measured scalar. Translation uses v*offset, while LOS
    directions rotate only, retaining their acquisition geometry.
    """
    if scan.time_offsets is None:
        raise ValueError("deskew requires per-point time_offsets (seconds from scan timestamp)")
    offsets = scan.time_offsets
    rotations = Rotation.from_rotvec(offsets[:, None] * angular_velocity)
    return Scan(rotations.apply(scan.points) + offsets[:, None] * velocity,
                scan.doppler, rotations.apply(scan.los), np.zeros(len(offsets)))


def dynamic_mask(scan, velocity, config):
    residual = scan.doppler + scan.los @ velocity
    threshold = config.tau0 + config.kappa * np.linalg.norm(scan.points, axis=1)
    return np.abs(residual) > threshold


def cluster_dynamics(points, config):
    if len(points) < max(config.min_cluster_size, config.min_samples + 1):
        return np.full(len(points), -1, dtype=int)
    import hdbscan

    return hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size,
                           min_samples=config.min_samples).fit_predict(points)


def well_conditioned(U, config):
    if len(U) < 3:
        return False
    singular = np.linalg.svd(U, compute_uv=False)
    return (singular[-1] >= config.min_singular_value
            and singular[0] / singular[-1] <= config.max_condition_number)


def reconstruct_velocities(scan, labels, ego_velocity, config):
    """Full 3D OLS and velocity consistency (Eqs. 5--8); no 1D/2D fallback."""
    labels = np.asarray(labels)
    if labels.shape != (len(scan.points),):
        raise ValueError("labels must match scan length")
    compensated = scan.doppler + scan.los @ ego_velocity
    accepted = np.zeros(len(labels), dtype=bool)
    velocities = {}
    for label in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == label)
        U, b = scan.los[indices], compensated[indices]
        if not well_conditioned(U, config):
            continue
        velocity = np.linalg.lstsq(U, b, rcond=None)[0]
        # The vector-valued v in the paper's threshold is interpreted as speed.
        threshold = config.velocity_relative_threshold * np.linalg.norm(velocity)
        inliers = np.abs(U @ velocity - b) <= max(threshold, 1e-12)
        if inliers.mean() < config.min_inlier_fraction:
            continue
        if not well_conditioned(U[inliers], config):
            continue
        velocities[int(label)] = velocity
        accepted[indices[inliers]] = True
    return velocities, accepted


def predict_source(scan, ego_velocity, dt, config):
    mask = dynamic_mask(scan, ego_velocity, config)
    labels = np.full(len(scan.points), -1, dtype=int)
    accepted = np.zeros(len(scan.points), dtype=bool)
    velocities = {}
    if config.ablation == "no_vf":
        mask[:] = False
    elif config.ablation != "no_dpp":
        indices = np.flatnonzero(mask)
        labels[indices] = cluster_dynamics(scan.points[indices], config)
        velocities, accepted = reconstruct_velocities(scan, labels, ego_velocity, config)
    predicted = scan.points.copy()
    for label, velocity in velocities.items():
        predicted[accepted & (labels == label)] += velocity * dt
    # Eq. 11: static set plus validated dynamics. Noise is excluded, not warped.
    keep = ~mask | accepted
    result = Scan(predicted[keep], scan.doppler[keep], scan.los[keep])
    statistics = {"points": len(scan.points), "dynamic_candidates": int(mask.sum()),
                  "clustered_dynamic": int((labels >= 0).sum()),
                  "predicted_dynamic": int(accepted.sum()), "clusters": len(velocities)}
    return result, statistics


def estimate_normals(points, config):
    """Local plane fitting. Invalid neighborhoods are excluded from matching."""
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(len(points), config.normal_max_nn),
                                     distance_upper_bound=config.normal_radius)
    neighbors = points[np.minimum(indices, len(points) - 1)]
    valid = np.isfinite(distances)
    count = valid.sum(axis=1)
    mean = np.sum(neighbors * valid[:, :, None], axis=1) / np.maximum(count[:, None], 1)
    centered = (neighbors - mean[:, None, :]) * valid[:, :, None]
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    normals[(count < 3) | (eigenvalues[:, 1] < 1e-12)] = 0
    return normals


def residuals_and_jacobians(points, source_los, source_doppler, target_points,
                            target_normals, target_los, target_doppler, transform):
    """Fixed-correspondence Eqs. 12--13, with [rotation, translation] columns."""
    rotated = points @ transform[:3, :3].T
    displacement = rotated + transform[:3, 3] - target_points
    geometry = np.einsum("ij,ij->i", target_normals, displacement)
    rays = (source_los * source_doppler[:, None]) @ transform[:3, :3].T
    doppler = np.einsum("ij,ij->i", target_los, rays) - target_doppler
    Jg = np.column_stack((np.cross(rotated, target_normals), target_normals))
    Jv = np.column_stack((np.cross(rays, target_los), np.zeros_like(rays)))
    return geometry, doppler, Jg, Jv


def tukey_weights(residual, scale):
    return np.maximum(0, 1 - (residual / scale) ** 2) ** 2


def displacement_residual(transform, displacement):
    """Optional Doppler displacement prior, additional to the paper objective."""
    R, t = transform[:3, :3], transform[:3, 3]
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return R.T @ t + displacement, np.column_stack((R.T @ skew, R.T))


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    converged: bool
    num_iterations: int
    fitness: float
    inlier_rmse: float
    status: str
    ego_velocity: np.ndarray
    statistics: Dict = field(default_factory=dict)


def register(source, target, dt, config=None, init_transform=None, ego_prior=None,
             rotation_prior=None):
    """Estimate T_target_source without GT, vehicle axes, or extrinsic calibration."""
    config = Config() if config is None else config
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    source, target = source.downsample(config.voxel_size), target.downsample(config.voxel_size)
    if min(len(source.points), len(target.points)) < 6:
        raise ValueError("At least six valid points per downsampled scan are required")
    velocity = estimate_ego_velocity(source, config, ego_prior)
    prior_weight = (config.ego_translation_weight
                    if np.linalg.norm(velocity) >= config.ego_prior_min_speed else 0.0)
    if config.deskew:
        omega = (np.zeros(3) if rotation_prior is None else
                 -Rotation.from_matrix(rotation_prior).as_rotvec() / dt)
        target_velocity = estimate_ego_velocity(target, config, velocity)
        source = deskew_scan(source, velocity, omega)
        target = deskew_scan(target, target_velocity, omega)
    static = ~dynamic_mask(source, velocity, config)
    information = source.los[static].T @ source.los[static] / max(len(source.points), 1)
    source, statistics = predict_source(source, velocity, dt, config)
    T = np.eye(4) if init_transform is None else np.array(init_transform, dtype=float, copy=True)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Initial transform must be a finite 4x4 matrix")
    if (not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1)):
        raise ValueError("Initial transform must belong to SE(3)")
    if init_transform is None:
        if config.use_rotation_prior and rotation_prior is not None:
            T[:3, :3] = rotation_prior
        T[:3, 3] = -T[:3, :3] @ velocity * dt
    normals = estimate_normals(target.points, config)
    lam = 0.0 if config.ablation == "no_dr" else config.lambda_doppler
    if config.backend == "cpp":
        if config.correspondence_range_scale != 0:
            raise ValueError("The C++ backend currently requires a fixed correspondence gate")
        from .native_backend import load_backend
        _native = load_backend()

        valid = np.linalg.norm(normals, axis=1) > 0
        target = target.select(valid)
        try:
            result = _native.register(source.points, source.doppler, source.los,
                                      target.points, target.doppler, target.los, normals[valid], T,
                                      config.max_correspondence_distance, lam,
                                      config.geometric_k, config.doppler_k, config.tolerance,
                                      config.max_iterations, prior_weight,
                                      velocity * dt, information)
        except RuntimeError as error:
            status = ("degenerate_system" if "degenerate_system" in str(error)
                      else "insufficient_correspondences" if "insufficient_correspondences" in str(error)
                      else None)
            if status is None:
                raise
            return RegistrationResult(T, False, 0, 0.0, float("inf"), status, velocity, statistics)
        return RegistrationResult(result["transformation"], result["converged"],
                                  result["iterations"], result["fitness"], result["rmse"],
                                  "converged" if result["converged"] else "max_iterations",
                                  velocity, statistics)
    tree = cKDTree(target.points)
    gates = (config.max_correspondence_distance + config.correspondence_range_scale
             * np.linalg.norm(source.points, axis=1))
    converged, status, iterations = False, "max_iterations", 0

    def correspondences():
        moved = source.points @ T[:3, :3].T + T[:3, 3]
        distance, idx = tree.query(moved)
        keep = (distance <= gates) & (np.linalg.norm(normals[idx], axis=1) > 0)
        return distance, idx, keep

    for iteration in range(config.max_iterations):
        distance, idx, keep = correspondences()
        if keep.sum() < 6:
            status = "insufficient_correspondences"
            break
        j = idx[keep]
        rg, rv, Jg, Jv = residuals_and_jacobians(
            source.points[keep], source.los[keep], source.doppler[keep],
            target.points[j], normals[j], target.los[j], target.doppler[j], T)
        wg = np.sqrt((1 - lam) * tukey_weights(rg, config.geometric_k))
        wv = np.sqrt(lam * tukey_weights(rv, config.doppler_k))
        A = np.vstack((Jg * wg[:, None], Jv * wv[:, None]))
        b = -np.concatenate((rg * wg, rv * wv))
        if prior_weight > 0:
            r_prior, J_prior = displacement_residual(T, velocity * dt)
            values, vectors = np.linalg.eigh(information)
            root_information = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
            root_information *= np.sqrt(prior_weight * keep.sum())
            A = np.vstack((A, root_information @ J_prior))
            b = np.r_[b, -root_information @ r_prior]
        delta, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 6:
            status = "degenerate_system"
            break
        old_fitness = float(keep.mean())
        old_rmse = float(np.sqrt(np.mean(distance[keep] ** 2)))
        T[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ T[:3, :3]
        T[:3, 3] += delta[3:]
        iterations = iteration + 1
        new_distance, _, new_keep = correspondences()
        # Open3D-style fitness/RMSE stopping, evaluated on the updated transform.
        stable = (new_keep.sum() >= 6
                  and abs(float(new_keep.mean()) - old_fitness) < config.tolerance
                  and abs(float(np.sqrt(np.mean(new_distance[new_keep] ** 2)))
                          - old_rmse) < config.tolerance)
        if stable:
            converged, status = True, "converged"
            break
    distance, _, keep = correspondences()
    fitness = float(keep.sum() / max(len(keep), 1))
    rmse = float(np.sqrt(np.mean(distance[keep] ** 2))) if keep.any() else float("inf")
    return RegistrationResult(T, converged, iterations, fitness, rmse, status,
                              velocity, statistics)
