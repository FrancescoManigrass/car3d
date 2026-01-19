from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DiffusionConfig


@dataclass(frozen=True)
class DiffusionJob:
    input_image: Path
    control_image: Path
    output_dir: Path
    denoise: float
    seed: int
    prompt: str
    negative_prompt: str
    controlnet_weight: float


def build_jobs(
    config: DiffusionConfig,
    input_images: Iterable[Path],
    control_images: Iterable[Path],
) -> list[DiffusionJob]:
    jobs: list[DiffusionJob] = []
    for input_image, control_image in zip(input_images, control_images):
        for denoise in config.denoise_levels:
            for seed in config.seeds:
                jobs.append(
                    DiffusionJob(
                        input_image=input_image,
                        control_image=control_image,
                        output_dir=config.output_dir,
                        denoise=denoise,
                        seed=seed,
                        prompt=config.prompt,
                        negative_prompt=config.negative_prompt,
                        controlnet_weight=config.controlnet_weight,
                    )
                )
    return jobs


def run_diffusion(jobs: Iterable[DiffusionJob], runner: callable | None = None) -> list[Path]:
    outputs: list[Path] = []
    for job in jobs:
        if runner is None:
            raise RuntimeError(
                "No diffusion runner provided. Hook this into SDXL/ComfyUI/etc."
            )
        outputs.extend(runner(job))
    return outputs
