from __future__ import annotations

import json
from pathlib import Path

from car3d_pipeline import (
    CameraConfig,
    DiffusionConfig,
    LoopConfig,
    RankingConfig,
    RenderConfig,
    Update3DConfig,
    run_loop,
)
from car3d_pipeline.config import ModelInput


def load_cameras(path: Path) -> list[CameraConfig]:
    data = json.loads(path.read_text())
    return [CameraConfig(**item) for item in data]


def main() -> None:
    base_dir = Path("/workspace/car3d")
    cameras = load_cameras(base_dir / "cameras.json")

    render = RenderConfig(
        blender_path=Path("/usr/bin/blender"),
        scene_path=base_dir / "scene.blend",
        output_dir=base_dir / "renders",
        cameras=cameras,
    )
    diffusion = DiffusionConfig(
        output_dir=base_dir / "diffusion",
        controlnet_mode="normals",
        denoise_levels=(0.25, 0.35, 0.45),
        prompt="sporty aggressive front fascia, clean automotive design",
        negative_prompt="deformed, asymmetry, text, logo",
        seeds=(42, 1337),
    )
    ranking = RankingConfig(
        view_weights={"front": 2.0, "front_three_quarter": 1.5},
    )
    update3d = Update3DConfig(
        mode="texture",
        output_dir=base_dir / "update3d",
        locked_groups=("wheels", "suspension"),
    )

    loop = LoopConfig(
        render=render,
        diffusion=diffusion,
        ranking=ranking,
        update3d=update3d,
        iterations=2,
        shortlist_k=5,
    )

    model = ModelInput(mesh_path=base_dir / "model.obj", hardpoints=[])

    results = run_loop(
        loop,
        model=model,
        render_script=base_dir / "scripts" / "blender_render.py",
        diffusion_runner=None,
    )

    for result in results:
        print(f"Iteration {result.iteration}: score={result.score}")


if __name__ == "__main__":
    main()
