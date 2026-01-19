import argparse

from src.run_loop import run_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blender", required=True, help="Path to blender binary (e.g. /usr/bin/blender)")
    p.add_argument("--mesh", required=True, help="Path to mesh OBJ with UVs")
    p.add_argument("--cameras", default="configs/cameras.json")
    p.add_argument("--hardpoints", default="configs/hardpoints.json")
    p.add_argument("--workdir", default="outputs/run1")

    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--negative",
        default="cartoon, deformed, extra wheels, text, logo, watermark, low quality",
    )
    p.add_argument("--control_type", default="normal", choices=["normal", "depth", "canny", "edges"])
    p.add_argument("--controlnet_model", default="diffusers/controlnet-canny-sdxl-1.0")
    p.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")

    p.add_argument("--denoise", type=float, default=0.4)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--iters_loop", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_loop(
        blender_bin=args.blender,
        mesh_path=args.mesh,
        cameras_json=args.cameras,
        hardpoints_json=args.hardpoints,
        work_dir=args.workdir,
        prompt=args.prompt,
        negative_prompt=args.negative,
        control_type=args.control_type,
        controlnet_model=args.controlnet_model,
        base_model=args.base_model,
        denoise_strength=args.denoise,
        num_candidates=args.num_candidates,
        iters_loop=args.iters_loop,
        device=args.device,
    )


if __name__ == "__main__":
    main()
