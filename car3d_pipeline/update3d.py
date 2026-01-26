from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Update3DConfig


@dataclass(frozen=True)
class UpdateResult:
    assets_path: Path
    notes: str


def update_texture_only(
    config: Update3DConfig,
    views: Iterable[Path],
) -> UpdateResult:
    """Stub for texture-only optimization."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return UpdateResult(
        assets_path=config.output_dir,
        notes="Texture-only optimization stub. Implement UV/texture fitting here.",
    )


def update_geometry_with_3dgs(
    config: Update3DConfig,
    views: Iterable[Path],
) -> UpdateResult:
    """Stub for 3D Gaussian Splatting optimization."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return UpdateResult(
        assets_path=config.output_dir,
        notes="3DGS optimization stub. Integrate gaussian splatting training here.",
    )


def run_update(config: Update3DConfig, views: Iterable[Path]) -> UpdateResult:
    if config.mode == "texture":
        return update_texture_only(config, views)
    if config.mode == "3dgs":
        return update_geometry_with_3dgs(config, views)
    raise ValueError(f"Unsupported update mode: {config.mode}")
