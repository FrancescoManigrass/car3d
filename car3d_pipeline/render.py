from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .config import CameraConfig, Hardpoint, RenderConfig


def project_points(
    points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
) -> np.ndarray:
    """Project 3D points into 2D using pinhole camera matrices."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be (N, 3) array.")
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homogenous = np.concatenate([points, ones], axis=1)
    camera_points = homogenous @ extrinsic.T
    camera_points = camera_points[:, :3]
    z = np.clip(camera_points[:, 2], 1e-6, None)
    normalized = camera_points / z[:, None]
    pixels = normalized @ intrinsic.T
    return pixels[:, :2]


def build_blender_command(config: RenderConfig, script_path: Path) -> list[str]:
    """Create a blender command line call for batch rendering."""
    return [
        str(config.blender_path),
        "-b",
        str(config.scene_path),
        "-P",
        str(script_path),
        "--",
        json.dumps({
            "output_dir": str(config.output_dir),
            "depth_near": config.depth_near,
            "depth_far": config.depth_far,
        }),
    ]


def write_camera_manifest(cameras: Sequence[CameraConfig], path: Path) -> None:
    path.write_text(json.dumps([asdict(camera) for camera in cameras], indent=2))


def generate_hardpoint_heatmap(
    image_size: tuple[int, int],
    hardpoints: Sequence[Hardpoint],
    camera: CameraConfig,
    sigma: float = 6.0,
) -> Image.Image:
    width, height = image_size
    points = np.array([hp.position for hp in hardpoints], dtype=np.float32)
    intrinsic = np.array(camera.intrinsic, dtype=np.float32)
    extrinsic = np.array(camera.extrinsic, dtype=np.float32)
    pixels = project_points(points, intrinsic, extrinsic)

    canvas = np.zeros((height, width), dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    for x, y in pixels:
        if np.isnan(x) or np.isnan(y):
            continue
        gaussian = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
        canvas = np.maximum(canvas, gaussian)

    canvas = np.clip(canvas / canvas.max() if canvas.max() > 0 else canvas, 0, 1)
    return Image.fromarray((canvas * 255).astype(np.uint8), mode="L")


def ensure_output_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def run_blender_render(config: RenderConfig, script_path: Path) -> None:
    ensure_output_dirs([config.output_dir])
    command = build_blender_command(config, script_path)
    subprocess.run(command, check=True)
