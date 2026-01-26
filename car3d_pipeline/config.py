from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int
    intrinsic: list[list[float]]
    extrinsic: list[list[float]]


@dataclass(frozen=True)
class RenderConfig:
    blender_path: Path
    scene_path: Path
    output_dir: Path
    cameras: Sequence[CameraConfig]
    depth_near: float = 0.1
    depth_far: float = 100.0


@dataclass(frozen=True)
class DiffusionConfig:
    output_dir: Path
    controlnet_mode: str
    denoise_levels: Sequence[float]
    prompt: str
    negative_prompt: str
    seeds: Sequence[int] = field(default_factory=lambda: [0])
    controlnet_weight: float = 0.8


@dataclass(frozen=True)
class RankingConfig:
    clip_weight: float = 0.6
    silhouette_weight: float = 0.25
    hardpoint_penalty_weight: float = 0.15
    view_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Update3DConfig:
    mode: str
    output_dir: Path
    locked_groups: Sequence[str] = field(default_factory=tuple)
    max_displacement: float = 0.0


@dataclass(frozen=True)
class LoopConfig:
    render: RenderConfig
    diffusion: DiffusionConfig
    ranking: RankingConfig
    update3d: Update3DConfig
    iterations: int
    shortlist_k: int


@dataclass(frozen=True)
class Hardpoint:
    name: str
    position: Iterable[float]
    group: str = "free"


@dataclass(frozen=True)
class ModelInput:
    mesh_path: Path
    hardpoints: Sequence[Hardpoint]
    locked_groups: Sequence[str] = field(default_factory=tuple)
