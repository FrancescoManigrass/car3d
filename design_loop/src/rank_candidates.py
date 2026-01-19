import numpy as np
import torch
import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .utils_geom import project_points


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def clip_ranker(device="cuda", model_name="openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    proc = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, proc


@torch.inference_mode()
def clip_score(model, proc, img: Image.Image, text: str, device="cuda"):
    inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    score = out.logits_per_image[0, 0].item()
    return score


def silhouette_iou(mask_a, mask_b):
    a = mask_a > 127
    b = mask_b > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + 1e-8
    return float(inter / union)


def estimate_candidate_mask(candidate_rgb, baseline_silhouette):
    base = baseline_silhouette > 127
    img = np.array(candidate_rgb.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = gray.copy()
    g[~base] = 0
    thr = int(np.percentile(g[base], 25)) if base.sum() > 0 else 50
    cand = (g > thr).astype(np.uint8) * 255
    cand = cv2.medianBlur(cand, 5)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cand


def hardpoint_patch_change(candidate_rgb, baseline_rgb, hp_uv, patch=21):
    c = np.array(candidate_rgb.convert("RGB")).astype(np.float32) / 255.0
    b = np.array(baseline_rgb.convert("RGB")).astype(np.float32) / 255.0
    height, width = c.shape[:2]
    radius = patch // 2
    pen = 0.0
    cnt = 0
    for (u, v) in hp_uv:
        u = int(round(u))
        v = int(round(v))
        if u < radius or v < radius or u >= width - radius or v >= height - radius:
            continue
        cp = c[v - radius : v + radius + 1, u - radius : u + radius + 1]
        bp = b[v - radius : v + radius + 1, u - radius : u + radius + 1]
        pen += float(np.mean(np.abs(cp - bp)))
        cnt += 1
    return pen / max(cnt, 1)


def rank_view_candidates(
    candidate_paths,
    baseline_rgb_path,
    baseline_sil_path,
    prompt_text,
    hardpoints_xyz,
    cam_def,
    width,
    height,
    weights=(0.6, 0.3, 0.1),
    device="cuda",
):
    w_clip, w_iou, w_hp = weights
    model, proc = clip_ranker(device=device)

    baseline_rgb = Image.open(baseline_rgb_path).convert("RGB")
    baseline_sil = load_gray(baseline_sil_path)

    pts = np.array([p["xyz"] for p in hardpoints_xyz], dtype=np.float32)
    u, v, _ = project_points(
        pts,
        cam_eye=cam_def["location"],
        cam_target=cam_def["look_at"],
        fov_deg=cam_def["fov_deg"],
        width=width,
        height=height,
    )
    hp_uv = list(zip(u.tolist(), v.tolist()))

    scored = []
    for path in candidate_paths:
        cand_rgb = Image.open(path).convert("RGB")

        cs = clip_score(model, proc, cand_rgb, prompt_text, device=device)

        cand_mask = estimate_candidate_mask(cand_rgb, baseline_sil)
        iou = silhouette_iou(baseline_sil, cand_mask)

        hp_pen = hardpoint_patch_change(cand_rgb, baseline_rgb, hp_uv, patch=21)

        total = w_clip * cs + w_iou * iou - w_hp * hp_pen
        scored.append(
            {
                "path": path,
                "clip": cs,
                "iou": iou,
                "hp_pen": hp_pen,
                "score": total,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
