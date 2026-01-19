import json
import os

import numpy as np
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2)


def load_image(path: str, mode="RGB"):
    img = Image.open(path)
    if mode is not None:
        img = img.convert(mode)
    return img


def save_image(img: Image.Image, path: str):
    ensure_dir(os.path.dirname(path))
    img.save(path)


def pil_to_np(img: Image.Image):
    return np.array(img)


def np_to_pil(arr: np.ndarray):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def list_views(render_root: str):
    views = []
    for name in sorted(os.listdir(render_root)):
        path = os.path.join(render_root, name)
        if not os.path.isdir(path):
            continue
        views.append(name)
    return views


def find_first_matching(path_dir: str, prefix: str, exts=(".png", ".exr")):
    for filename in sorted(os.listdir(path_dir)):
        if filename.startswith(prefix) and filename.lower().endswith(exts):
            return os.path.join(path_dir, filename)
    raise FileNotFoundError(f"Missing {prefix}* in {path_dir}")
