import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True, help="Path to mesh (OBJ/FBX/GLB etc.)")
    p.add_argument("--cameras", required=True, help="Path to cameras.json")
    p.add_argument("--hardpoints", required=True, help="Path to hardpoints.json")
    p.add_argument("--out", required=True, help="Output directory")
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if bpy.context.preferences.addons.get("cycles") else "CPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    return scene


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects imported.")
    bpy.context.view_layer.objects.active = meshes[0]
    for obj in meshes:
        obj.select_set(True)
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "MODEL"
    return obj


def setup_light_and_material(obj):
    bpy.ops.object.light_add(type="AREA", location=(4, -4, 6))
    light_a = bpy.context.object
    light_a.data.energy = 1500
    light_a.data.size = 5

    bpy.ops.object.light_add(type="AREA", location=(-4, 4, 6))
    light_b = bpy.context.object
    light_b.data.energy = 1200
    light_b.data.size = 5

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 8))
    light_c = bpy.context.object
    light_c.data.energy = 800
    light_c.data.size = 8

    mat = bpy.data.materials.new(name="NeutralMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Metallic"].default_value = 0.1

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def look_at(obj, target):
    direction = Vector(target) - obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def setup_compositor(out_dir, base_name):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rlayers = tree.nodes.new("CompositorNodeRLayers")

    def file_out(label, path):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.label = label
        node.base_path = out_dir
        node.file_slots[0].path = path
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        return node

    rgb = file_out("rgb", f"{base_name}/rgb_")
    tree.links.new(rlayers.outputs["Image"], rgb.inputs[0])

    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.label = "depth"
    depth.base_path = out_dir
    depth.file_slots[0].path = f"{base_name}/depth_"
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_mode = "BW"
    tree.links.new(rlayers.outputs["Depth"], depth.inputs[0])

    normal = file_out("normal", f"{base_name}/normal_")
    tree.links.new(rlayers.outputs["Normal"], normal.inputs[0])

    scene.view_layers["ViewLayer"].use_pass_object_index = True
    idmask = tree.nodes.new("CompositorNodeIDMask")
    idmask.index = 1
    tree.links.new(rlayers.outputs["IndexOB"], idmask.inputs["ID value"])
    sil = file_out("silhouette", f"{base_name}/sil_")
    tree.links.new(idmask.outputs["Alpha"], sil.inputs[0])


def ensure_passes():
    scene = bpy.context.scene
    view_layer = scene.view_layers["ViewLayer"]
    view_layer.use_pass_normal = True
    view_layer.use_pass_z = True


def set_object_index(obj, idx=1):
    obj.pass_index = idx


def bpy_extras_object_utils_world_to_camera_view(scene, cam, coord):
    co_local = cam.matrix_world.normalized().inverted() @ Vector(coord)
    z = -co_local.z
    if z <= 0.0:
        return Vector((0.0, 0.0, -1.0))
    frame = cam.data.view_frame(scene=scene)
    left = frame[0].x
    right = frame[1].x
    bottom = frame[0].y
    top = frame[2].y
    x = (co_local.x / z - left) / (right - left)
    y = (co_local.y / z - bottom) / (top - bottom)
    return Vector((x, y, z))


def project_point_simple(cam, point, width, height):
    co_ndc = bpy_extras_object_utils_world_to_camera_view(bpy.context.scene, cam, point)
    if co_ndc.z < 0:
        return 0, 0, -1
    u = int(co_ndc.x * width)
    v = int((1.0 - co_ndc.y) * height)
    return u, v, co_ndc.z


def add_gaussian(img, cx, cy, sigma):
    height, width = img.shape
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return
    radius = int(3 * sigma)
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    xx, yy = np.meshgrid(xs, ys)
    gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], gauss.astype(np.float32))


def save_hardpoints_heatmap(cam, hardpoints, out_path, width, height, sigma_px):
    canvas = np.zeros((height, width), dtype=np.float32)
    for hp in hardpoints:
        x, y, z = hp["xyz"]
        u, v, depth = project_point_simple(cam, (x, y, z), width, height)
        if depth < 0:
            continue
        add_gaussian(canvas, u, v, sigma_px)
    canvas = np.clip(canvas, 0.0, 1.0)
    img = bpy.data.images.new("hardpoints", width=width, height=height, alpha=True, float_buffer=True)
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., 0] = canvas
    rgba[..., 1] = canvas
    rgba[..., 2] = canvas
    rgba[..., 3] = 1.0
    img.pixels = rgba.flatten()
    img.filepath_raw = out_path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.cameras, "r") as handle:
        cam_cfg = json.load(handle)
    with open(args.hardpoints, "r") as handle:
        hp_cfg = json.load(handle)

    scene = reset_scene()
    scene.render.resolution_x = int(cam_cfg["image_width"])
    scene.render.resolution_y = int(cam_cfg["image_height"])
    scene.render.resolution_percentage = 100
    ensure_passes()

    obj = import_mesh(args.mesh)
    setup_light_and_material(obj)
    set_object_index(obj, 1)

    bpy.ops.object.camera_add(location=(0, -5, 1.5))
    cam = bpy.context.object
    scene.camera = cam

    for cam_def in cam_cfg["cameras"]:
        name = cam_def["name"]
        cam.location = Vector(cam_def["location"])
        look_at(cam, cam_def["look_at"])
        cam.data.lens_unit = "FOV"
        cam.data.angle = math.radians(float(cam_def["fov_deg"]))

        setup_compositor(args.out, name)

        bpy.ops.render.render(write_still=False)

        hp_dir = os.path.join(args.out, name)
        os.makedirs(hp_dir, exist_ok=True)
        hp_path = os.path.join(hp_dir, "hardpoints.png")
        save_hardpoints_heatmap(
            cam,
            hp_cfg["points"],
            hp_path,
            scene.render.resolution_x,
            scene.render.resolution_y,
            int(hp_cfg.get("sigma_px", 16)),
        )

    print("Done rendering maps.")


if __name__ == "__main__":
    main()
