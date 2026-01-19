import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
)

from .utils_io import ensure_dir


class VGGFeatures(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).to(device).eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.slice(x)


def pil_to_tensor(img: Image.Image, device="cuda", size=None):
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return t


def look_at_view_transform(eye, at, up):
    z = F.normalize(eye - at, dim=1)
    x = F.normalize(torch.cross(up, z, dim=1), dim=1)
    y = torch.cross(z, x, dim=1)
    R = torch.stack([x, y, z], dim=1)
    T = -torch.bmm(R, eye.unsqueeze(-1)).squeeze(-1)
    return R, T


def optimize_texture(
    mesh_obj_path: str,
    target_images_by_view: dict,
    cam_defs_by_view: dict,
    out_dir: str,
    tex_res: int = 1024,
    image_size: int = 512,
    iters: int = 300,
    lr: float = 0.08,
    device="cuda",
):
    ensure_dir(out_dir)

    mesh = load_objs_as_meshes([mesh_obj_path], device=device)

    if mesh.textures is None or not isinstance(mesh.textures, TexturesUV):
        raise RuntimeError(
            "Mesh has no UV textures. Please unwrap UVs and export OBJ with vt coordinates."
        )

    tex = torch.rand((1, tex_res, tex_res, 3), device=device, requires_grad=True)

    faces_uvs = mesh.textures.faces_uvs_padded()
    verts_uvs = mesh.textures.verts_uvs_padded()

    vgg = VGGFeatures(device=device)

    view_names = list(target_images_by_view.keys())
    targets = []
    cam_defs = []
    for view_name in view_names:
        target = pil_to_tensor(
            Image.open(target_images_by_view[view_name]).convert("RGB"),
            device=device,
            size=(image_size, image_size),
        )
        targets.append(target)
        cam_defs.append(cam_defs_by_view[view_name])
    targets = torch.cat(targets, dim=0)

    Rs, Ts, fovs = [], [], []
    for cam_def in cam_defs:
        eye = torch.tensor(cam_def["location"], dtype=torch.float32, device=device).unsqueeze(0)
        at = torch.tensor(cam_def["look_at"], dtype=torch.float32, device=device).unsqueeze(0)
        up = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
        R, T = look_at_view_transform(eye=eye, at=at, up=up)
        Rs.append(R)
        Ts.append(T)
        fovs.append(cam_def["fov_deg"])
    R = torch.cat(Rs, dim=0)
    T = torch.cat(Ts, dim=0)
    fov = torch.tensor(fovs, dtype=torch.float32, device=device)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0.0, -3.0, 5.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    opt = torch.optim.Adam([tex], lr=lr)

    for it in range(iters):
        opt.zero_grad(set_to_none=True)

        textures = TexturesUV(
            maps=tex,
            faces_uvs=faces_uvs,
            verts_uvs=verts_uvs,
        )
        mesh_t = mesh.clone()
        mesh_t.textures = textures

        rendered = renderer(mesh_t)
        rgb = rendered[..., :3].permute(0, 3, 1, 2).contiguous()

        l1 = (rgb - targets).abs().mean()

        f1 = vgg(rgb)
        f2 = vgg(targets)
        perc = (f1 - f2).abs().mean()

        loss = 1.0 * l1 + 0.4 * perc
        loss.backward()
        opt.step()

        with torch.no_grad():
            tex.clamp_(0.0, 1.0)

        if (it + 1) % 50 == 0:
            print(
                f"[texopt] it={it + 1}/{iters} "
                f"loss={loss.item():.4f} l1={l1.item():.4f} perc={perc.item():.4f}"
            )

    tex_img = (tex.detach().cpu().numpy()[0] * 255.0).astype(np.uint8)
    Image.fromarray(tex_img).save(os.path.join(out_dir, "optimized_texture.png"))

    return os.path.join(out_dir, "optimized_texture.png")
