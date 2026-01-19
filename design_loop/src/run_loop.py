import os
import subprocess

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

from .diffusion_generate import (
    build_sdxl_controlnet_pipe,
    generate_candidates_for_view,
    load_control_image,
)
from .rank_candidates import rank_view_candidates
from .texture_optimize import optimize_texture
from .utils_io import ensure_dir, find_first_matching, list_views, read_json, save_json


def exr_depth_to_png(exr_path: str, png_path: str):
    d = iio.imread(exr_path)
    if d.ndim == 3:
        d = d[..., 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = d[d > 1e-6]
    if nonzero.size == 0:
        norm = np.zeros_like(d, dtype=np.float32)
    else:
        lo = np.percentile(nonzero, 2)
        hi = np.percentile(nonzero, 98)
        norm = (d - lo) / (hi - lo + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)
    norm = 1.0 - norm
    rgb = (np.stack([norm, norm, norm], axis=-1) * 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(png_path)


def run_blender_render(blender_bin, mesh_path, cameras_json, hardpoints_json, out_dir):
    ensure_dir(out_dir)
    cmd = [
        blender_bin,
        "--background",
        "--python",
        os.path.join("blender", "render_maps.py"),
        "--",
        "--mesh",
        mesh_path,
        "--cameras",
        cameras_json,
        "--hardpoints",
        hardpoints_json,
        "--out",
        out_dir,
    ]
    print("Running Blender:", " ".join(cmd))
    subprocess.check_call(cmd)


def run_loop(
    blender_bin: str,
    mesh_path: str,
    cameras_json: str,
    hardpoints_json: str,
    work_dir: str,
    prompt: str,
    negative_prompt: str,
    control_type: str = "normal",
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0",
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    denoise_strength: float = 0.4,
    num_candidates: int = 8,
    iters_loop: int = 3,
    device: str = "cuda",
):
    ensure_dir(work_dir)
    cams = read_json(cameras_json)
    hps = read_json(hardpoints_json)

    renders_root = os.path.join(work_dir, "renders")
    cands_root = os.path.join(work_dir, "candidates")
    chosen_root = os.path.join(work_dir, "chosen")
    tex_root = os.path.join(work_dir, "textures")

    ensure_dir(renders_root)
    ensure_dir(cands_root)
    ensure_dir(chosen_root)
    ensure_dir(tex_root)

    run_blender_render(blender_bin, mesh_path, cameras_json, hardpoints_json, renders_root)

    views = list_views(renders_root)
    for view in views:
        vdir = os.path.join(renders_root, view)
        depth_exr = find_first_matching(vdir, "depth_", exts=(".exr",))
        depth_png = os.path.join(vdir, "depth.png")
        exr_depth_to_png(depth_exr, depth_png)

        rgb = Image.open(find_first_matching(vdir, "rgb_", exts=(".png",))).convert("RGB")
        import cv2

        arr = np.array(rgb)
        edges = cv2.Canny(arr, 100, 200)
        edges = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)
        Image.fromarray(edges).save(os.path.join(vdir, "edges.png"))

    pipe = build_sdxl_controlnet_pipe(
        base_model=base_model,
        controlnet_model=controlnet_model,
        torch_dtype=(__import__("torch").float16 if device.startswith("cuda") else __import__("torch").float32),
        device=device,
    )

    cam_defs_by_view = {c["name"]: c for c in cams["cameras"]}
    width = int(cams["image_width"])
    height = int(cams["image_height"])

    history = []

    for loop_i in range(iters_loop):
        print(f"\n===== LOOP {loop_i + 1}/{iters_loop} =====")

        chosen_by_view = {}

        for view in tqdm(views, desc="views"):
            vdir = os.path.join(renders_root, view)
            rgb_path = find_first_matching(vdir, "rgb_", exts=(".png",))
            normal_path = find_first_matching(vdir, "normal_", exts=(".png",))
            sil_path = find_first_matching(vdir, "sil_", exts=(".png",))

            paths = {
                "rgb": rgb_path,
                "normal": normal_path,
                "sil": sil_path,
                "depth_png": os.path.join(vdir, "depth.png"),
                "edges": os.path.join(vdir, "edges.png"),
                "hardpoints": os.path.join(vdir, "hardpoints.png"),
            }

            ctrl = load_control_image(control_type, paths)

            out_v = os.path.join(cands_root, f"loop_{loop_i:02d}", view)
            cand_paths = generate_candidates_for_view(
                pipe=pipe,
                rgb_path=rgb_path,
                control_img=ctrl,
                out_dir=out_v,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_candidates=num_candidates,
                denoise_strength=denoise_strength,
                controlnet_conditioning_scale=0.9,
                guidance_scale=6.5,
                steps=25,
                seed=1000 * (loop_i + 1),
            )

            ranked = rank_view_candidates(
                candidate_paths=cand_paths,
                baseline_rgb_path=rgb_path,
                baseline_sil_path=sil_path,
                prompt_text=prompt,
                hardpoints_xyz=hps["points"],
                cam_def=cam_defs_by_view[view],
                width=width,
                height=height,
                weights=(0.6, 0.3, 0.1),
                device=device,
            )
            best = ranked[0]
            chosen_by_view[view] = best["path"]

            save_json(ranked, os.path.join(out_v, "ranking.json"))

            ensure_dir(os.path.join(chosen_root, f"loop_{loop_i:02d}"))
            Image.open(best["path"]).save(
                os.path.join(chosen_root, f"loop_{loop_i:02d}", f"{view}.png")
            )

        tex_out_dir = os.path.join(tex_root, f"loop_{loop_i:02d}")
        ensure_dir(tex_out_dir)

        target_images = {
            view: os.path.join(chosen_root, f"loop_{loop_i:02d}", f"{view}.png")
            for view in views
        }
        cam_defs = {view: cam_defs_by_view[view] for view in views}

        tex_path = optimize_texture(
            mesh_obj_path=mesh_path,
            target_images_by_view=target_images,
            cam_defs_by_view=cam_defs,
            out_dir=tex_out_dir,
            tex_res=1024,
            image_size=512,
            iters=300,
            lr=0.08,
            device=device,
        )

        history.append(
            {
                "loop": loop_i,
                "chosen_images": target_images,
                "optimized_texture": tex_path,
            }
        )
        save_json(history, os.path.join(work_dir, "history.json"))

        print(f"[loop {loop_i}] texture saved:", tex_path)
        print(
            "NOTE: apply optimized_texture.png to your mesh material in DCC "
            "for next loop if you want fully closed loop.\n"
        )

    print("DONE. History:", os.path.join(work_dir, "history.json"))
