import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

from .utils_io import ensure_dir


def make_canny(pil_img: Image.Image, low=100, high=200):
    arr = np.array(pil_img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def load_control_image(control_type: str, paths: dict):
    if control_type == "canny":
        return make_canny(Image.open(paths["rgb"]).convert("RGB"))
    if control_type == "normal":
        return Image.open(paths["normal"]).convert("RGB")
    if control_type == "depth":
        return Image.open(paths["depth_png"]).convert("RGB")
    if control_type == "edges":
        return Image.open(paths["edges"]).convert("RGB")
    raise ValueError(f"Unknown control_type: {control_type}")


def build_sdxl_controlnet_pipe(
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    controlnet_model="diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    device="cuda",
):
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch_dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipe.to(device)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


@torch.inference_mode()
def generate_candidates_for_view(
    pipe,
    rgb_path: str,
    control_img: Image.Image,
    out_dir: str,
    prompt: str,
    negative_prompt: str,
    num_candidates: int = 8,
    denoise_strength: float = 0.4,
    controlnet_conditioning_scale: float = 0.9,
    guidance_scale: float = 6.5,
    steps: int = 25,
    seed: int = 0,
):
    ensure_dir(out_dir)
    init_image = Image.open(rgb_path).convert("RGB")

    generator = torch.Generator(device=pipe.device)
    results = []
    for idx in range(num_candidates):
        generator.manual_seed(seed + idx)
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=control_img,
            strength=denoise_strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        path = os.path.join(out_dir, f"cand_{idx:02d}.png")
        img.save(path)
        results.append(path)
    return results
