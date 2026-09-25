"""Interactive four-stage viewer for the included HeRCULES demo."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .core import (
    Config,
    Scan,
    cluster_dynamics,
    dynamic_mask,
    estimate_ego_velocity,
    reconstruct_velocities,
    register,
)
from .io import load_scan, scan_paths


@dataclass
class MethodStage:
    scan: Scan
    ego_velocity: np.ndarray
    dynamic: np.ndarray
    labels: np.ndarray
    accepted: np.ndarray
    velocities: dict
    predicted_points: np.ndarray
    keep: np.ndarray


def analyze_frame(scan, dt, config):
    """Expose the same velocity-filter, reconstruction, and prediction stages as register()."""
    scan = scan.downsample(config.voxel_size)
    ego_velocity = estimate_ego_velocity(scan, config)
    dynamic = dynamic_mask(scan, ego_velocity, config)
    labels = np.full(len(scan.points), -1, dtype=int)
    accepted = np.zeros(len(scan.points), dtype=bool)
    velocities = {}
    if config.ablation == "no_vf":
        dynamic[:] = False
    elif config.ablation != "no_dpp":
        indices = np.flatnonzero(dynamic)
        labels[indices] = cluster_dynamics(scan.points[indices], config)
        velocities, accepted = reconstruct_velocities(
            scan, labels, ego_velocity, config
        )
    predicted = scan.points.copy()
    for label, velocity in velocities.items():
        predicted[accepted & (labels == label)] += velocity * dt
    keep = ~dynamic | accepted
    return MethodStage(
        scan, ego_velocity, dynamic, labels, accepted, velocities, predicted, keep
    )


def choose_backend(requested):
    if requested != "auto":
        return requested
    try:
        from .native_backend import load_backend

        load_backend()
        return "cpp"
    except (ImportError, RuntimeError, OSError):
        return "numpy"


class DemoViewer:
    def __init__(self, directory, config, backend, view_range, point_size,
                 max_correspondences, initial_frame=0, interval_ms=700):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.plt = plt
        self.directory = Path(directory)
        self.paths = scan_paths(self.directory / "scans", "npz")
        if len(self.paths) < 2:
            raise ValueError("The demo needs at least two scans")
        self.timestamps = np.asarray([float(path.stem) * 1e-9 for path in self.paths])
        self.dts = np.diff(self.timestamps)
        if np.any(self.dts <= 0):
            raise ValueError("Demo scan timestamps must be strictly increasing")
        self.scans = [load_scan(path, "npz") for path in self.paths]
        last_dt = float(np.median(self.dts))
        stage_dts = np.r_[self.dts, last_dt]
        self.stages = [
            analyze_frame(scan, float(dt), config)
            for scan, dt in zip(self.scans, stage_dts)
        ]
        self.config = replace(config, backend=backend)
        self.backend = backend
        self.view_range = view_range
        self.point_size = point_size
        self.max_correspondences = max_correspondences
        self.registration_cache = {}
        metadata_path = self.directory / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}

        self.figure, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#10151c")
        self.axes = axes.ravel()
        self.figure.subplots_adjust(bottom=0.12, top=0.90, wspace=0.08, hspace=0.17)
        slider_axis = self.figure.add_axes([0.16, 0.035, 0.68, 0.025], facecolor="#27313d")
        self.slider = Slider(
            slider_axis, "pair", 0, len(self.paths) - 2,
            valinit=min(max(initial_frame, 0), len(self.paths) - 2), valstep=1,
            color="#00c2a8",
        )
        self.slider.on_changed(lambda value: self.draw(int(value)))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.figure.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.advance)
        self.playing = False
        self.draw(int(self.slider.val))

    def registration(self, index):
        if index not in self.registration_cache:
            print(f"Registering pair {index + 1}/{len(self.paths) - 1} ({self.backend})...", flush=True)
            self.registration_cache[index] = register(
                self.scans[index], self.scans[index + 1], float(self.dts[index]), self.config
            )
        return self.registration_cache[index]

    def visible(self, points):
        r = self.view_range
        return (np.abs(points[:, 0]) <= r) & (np.abs(points[:, 1]) <= r)

    def setup_axis(self, axis, title):
        axis.clear()
        axis.set_facecolor("#10151c")
        axis.set_title(title, color="white", fontsize=11, fontweight="bold")
        axis.set_xlim(-self.view_range, self.view_range)
        axis.set_ylim(-self.view_range, self.view_range)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#344252", linewidth=0.4, alpha=0.35)
        axis.tick_params(colors="#9eabb8", labelsize=8)
        axis.set_xlabel("x [m]", color="#9eabb8")
        axis.set_ylabel("y [m]", color="#9eabb8")
        axis.scatter([0], [0], marker="^", s=45, c="#ffffff", edgecolors="#10151c", zorder=8)

    def scatter(self, axis, points, **kwargs):
        mask = self.visible(points)
        return axis.scatter(points[mask, 0], points[mask, 1], **kwargs)

    def draw(self, index):
        index = int(np.clip(index, 0, len(self.paths) - 2))
        stage = self.stages[index]
        target = self.scans[index + 1].downsample(self.config.voxel_size)
        result = self.registration(index)
        dt = float(self.dts[index])

        raw, classified, prediction, matching = self.axes
        self.setup_axis(raw, "1  Raw Doppler")
        visible = self.visible(stage.scan.points)
        colors = np.clip(stage.scan.doppler[visible], -5.0, 5.0)
        scatter = raw.scatter(
            stage.scan.points[visible, 0], stage.scan.points[visible, 1],
            c=colors, cmap="coolwarm", vmin=-5, vmax=5,
            s=self.point_size, linewidths=0, alpha=0.85,
        )
        if not hasattr(self, "colorbar"):
            self.colorbar = self.figure.colorbar(scatter, ax=raw, fraction=0.035, pad=0.02)
            self.colorbar.set_label("radial velocity [m/s]", color="white")
            self.colorbar.ax.tick_params(colors="white", labelsize=8)

        self.setup_axis(classified, "2  Velocity filter and validated clusters")
        static = ~stage.dynamic
        rejected = stage.dynamic & ~stage.accepted
        self.scatter(classified, stage.scan.points[static], s=self.point_size, c="#627384", alpha=0.35, linewidths=0, label="static")
        self.scatter(classified, stage.scan.points[rejected], s=self.point_size * 1.7, c="#ff9f43", alpha=0.75, linewidths=0, label="candidate / rejected")
        self.scatter(classified, stage.scan.points[stage.accepted], s=self.point_size * 2.0, c="#00d084", alpha=0.9, linewidths=0, label="validated dynamic")
        classified.legend(loc="lower right", fontsize=7, facecolor="#17202a", labelcolor="white")

        self.setup_axis(prediction, "3  Constant-velocity dynamic prediction")
        self.scatter(prediction, stage.scan.points[static], s=self.point_size, c="#627384", alpha=0.25, linewidths=0)
        self.scatter(prediction, stage.scan.points[stage.accepted], s=self.point_size * 1.8, facecolors="none", edgecolors="#ff9f43", alpha=0.65, linewidths=0.45, label="measured")
        self.scatter(prediction, stage.predicted_points[stage.accepted], s=self.point_size * 2.0, c="#00d084", alpha=0.9, linewidths=0, label="predicted")
        for label, velocity in stage.velocities.items():
            selected = stage.accepted & (stage.labels == label)
            if not selected.any():
                continue
            center = stage.scan.points[selected].mean(axis=0)
            displacement = velocity * dt
            if self.visible(center[None])[0]:
                prediction.arrow(center[0], center[1], displacement[0], displacement[1],
                                 color="#f7dc6f", width=0.025, head_width=0.32,
                                 length_includes_head=True, zorder=7)
        prediction.legend(loc="lower right", fontsize=7, facecolor="#17202a", labelcolor="white")

        self.setup_axis(matching, "4  Predicted source aligned to target")
        T = result.transformation
        source_points = stage.predicted_points[stage.keep] @ T[:3, :3].T + T[:3, 3]
        source_dynamic = stage.accepted[stage.keep]
        self.scatter(matching, target.points, s=self.point_size, c="#85929e", alpha=0.32, linewidths=0, label="target")
        self.scatter(matching, source_points[~source_dynamic], s=self.point_size, c="#3498db", alpha=0.38, linewidths=0, label="aligned source")
        self.scatter(matching, source_points[source_dynamic], s=self.point_size * 2.0, c="#ff4f9a", alpha=0.88, linewidths=0, label="predicted dynamic")
        distances, nearest = cKDTree(target.points).query(source_points)
        matches = np.flatnonzero((distances <= self.config.max_correspondence_distance) & self.visible(source_points))
        if len(matches) > self.max_correspondences:
            matches = matches[np.linspace(0, len(matches) - 1, self.max_correspondences, dtype=int)]
        for source_index in matches:
            a, b = source_points[source_index], target.points[nearest[source_index]]
            matching.plot([a[0], b[0]], [a[1], b[1]], color="#86f7d0", linewidth=0.35, alpha=0.35)
        matching.legend(loc="lower right", fontsize=7, facecolor="#17202a", labelcolor="white")

        self.figure.suptitle(
            "Dynamic-ICP · HeRCULES River Island\n"
            f"pair {index + 1:02d}/{len(self.paths) - 1}  ·  Δt={dt:.3f}s  ·  "
            f"ego={np.linalg.norm(stage.ego_velocity):.2f}m/s  ·  "
            f"dynamic={stage.dynamic.sum()}  ·  predicted={stage.accepted.sum()}  ·  "
            f"clusters={len(stage.velocities)}  ·  {result.status}, "
            f"iter={result.num_iterations}, fitness={result.fitness:.3f}  ·  backend={self.backend}",
            color="white", fontsize=12, fontweight="bold",
        )
        self.figure.canvas.draw_idle()

    def on_key(self, event):
        if event.key == " ":
            self.set_playing(not self.playing)
        elif event.key in ("right", "d"):
            self.slider.set_val(min(int(self.slider.val) + 1, len(self.paths) - 2))
        elif event.key in ("left", "a"):
            self.slider.set_val(max(int(self.slider.val) - 1, 0))
        elif event.key == "home":
            self.slider.set_val(0)
        elif event.key == "end":
            self.slider.set_val(len(self.paths) - 2)

    def advance(self):
        next_index = (int(self.slider.val) + 1) % (len(self.paths) - 1)
        self.slider.set_val(next_index)

    def set_playing(self, playing):
        self.playing = bool(playing)
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()


def load_config(path):
    values = json.loads(Path(path).read_text())
    return Config(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset/hercules_dynamic"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/datasets/reference.json")
    )
    parser.add_argument("--backend", choices=("auto", "numpy", "cpp"), default="auto")
    parser.add_argument("--frame", type=int, default=0, help="Initial pair index")
    parser.add_argument("--view-range", type=float, default=35.0)
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--max-correspondences", type=int, default=100)
    parser.add_argument("--play", action="store_true", help="Start interactive playback immediately")
    parser.add_argument("--interval-ms", type=int, default=700, help="Interactive playback interval")
    parser.add_argument("--save", type=Path, help="Save one PNG or the complete GIF instead of opening a window")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if (args.view_range <= 0 or args.point_size <= 0
            or args.max_correspondences < 0 or args.interval_ms < 1):
        parser.error("view range/point size must be positive and correspondences nonnegative")
    if args.save is not None:
        if args.save.exists():
            parser.error(f"refusing to overwrite {args.save}")
        import matplotlib
        matplotlib.use("Agg")

    backend = choose_backend(args.backend)
    print(f"Viewer registration backend: {backend}", flush=True)
    viewer = DemoViewer(
        args.dataset, load_config(args.config), backend, args.view_range,
        args.point_size, args.max_correspondences, args.frame, args.interval_ms,
    )
    if args.save is None:
        print("Controls: Space play/pause, Left/A previous, Right/D next, Home/End jump", flush=True)
        if args.play:
            viewer.set_playing(True)
        viewer.plt.show()
        return 0

    args.save.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.save.suffix.lower()
    if suffix == ".png":
        viewer.figure.savefig(args.save, dpi=args.dpi, facecolor=viewer.figure.get_facecolor())
    elif suffix == ".gif":
        from matplotlib.animation import FuncAnimation, PillowWriter

        animation = FuncAnimation(
            viewer.figure,
            lambda index: viewer.slider.set_val(index),
            frames=range(len(viewer.paths) - 1),
            interval=1000 / args.fps,
            repeat=True,
        )
        animation.save(args.save, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    else:
        parser.error("--save must end in .png or .gif")
    print(f"Saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
