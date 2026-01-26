from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import LoopConfig, ModelInput
from .diffusion import build_jobs, run_diffusion
from .ranking import aggregate_scores
from .render import ensure_output_dirs
from .update3d import run_update


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    score: float
    assets_path: Path


def run_loop(
    config: LoopConfig,
    model: ModelInput,
    render_script: Path,
    diffusion_runner: callable | None = None,
) -> Sequence[IterationResult]:
    ensure_output_dirs([config.render.output_dir, config.diffusion.output_dir])
    results: list[IterationResult] = []

    for iteration in range(config.iterations):
        # Rendering is expected to happen via Blender script; call externally.
        # Placeholder: assume renders are already in output_dir.
        rgb_images = sorted(config.render.output_dir.glob("*_rgb.png"))
        control_images = sorted(config.render.output_dir.glob("*_control.png"))

        jobs = build_jobs(config.diffusion, rgb_images, control_images)
        generated = run_diffusion(jobs, diffusion_runner)

        # Placeholder scoring aggregation; integrate CLIP + silhouette IOU here.
        score = aggregate_scores([], config.ranking)

        shortlisted = generated[: config.shortlist_k]
        update_result = run_update(config.update3d, shortlisted)

        results.append(
            IterationResult(
                iteration=iteration,
                score=score,
                assets_path=update_result.assets_path,
            )
        )

    return results
