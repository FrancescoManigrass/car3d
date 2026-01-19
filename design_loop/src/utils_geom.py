import numpy as np


def normalize(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n


def look_at_matrix(eye, target, up=np.array([0, 0, 1], dtype=np.float32)):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    up_vec = np.cross(right, forward)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = right
    mat[:3, 1] = up_vec
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def world_to_camera(mat_c2w):
    return np.linalg.inv(mat_c2w)


def project_points(points_xyz, cam_eye, cam_target, fov_deg, width, height):
    """
    Returns pixel coords (u,v) and depth (z_cam positive).
    """
    mat_c2w = look_at_matrix(cam_eye, cam_target)
    mat_w2c = world_to_camera(mat_c2w)

    pts = np.concatenate([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float32)], axis=1)
    pc = (mat_w2c @ pts.T).T
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    depth = -z
    depth = np.maximum(depth, 1e-6)

    fov = np.deg2rad(fov_deg)
    fy = 0.5 * height / np.tan(0.5 * fov)
    fx = fy

    u = fx * (x / depth) + width * 0.5
    v = -fy * (y / depth) + height * 0.5
    return u, v, depth
