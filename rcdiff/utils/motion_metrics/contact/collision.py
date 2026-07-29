import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


TORCH_MESH_ISECT_ROOT = Path(__file__).resolve().parent / "torch_mesh_isect"
if str(TORCH_MESH_ISECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCH_MESH_ISECT_ROOT))


def numpy2set(arr):
    return set(tuple(x) for x in arr)


def compute_vertex_normals_efficient(vertices, faces):
    face_vertices = vertices[:, faces.flatten(), :].reshape(vertices.shape[0], faces.shape[0], 3, 3)
    v1, v2, v3 = np.split(face_vertices, 3, axis=2)
    edge1 = v2.squeeze(axis=2) - v1.squeeze(axis=2)
    edge2 = v3.squeeze(axis=2) - v1.squeeze(axis=2)
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.linalg.norm(face_normals, axis=2, keepdims=True) + 1e-5

    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for j in range(3):
            vertex_normals[:, face[j], :] += face_normals[:, i, :]
    vertex_normals /= np.linalg.norm(vertex_normals, axis=2, keepdims=True) + 1e-5
    return vertex_normals


def inflate_mesh_efficient(vertices, vertex_normals, distance=0.01):
    return vertices + vertex_normals * distance


def _load_smplx_params(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    return data


def get_smplx_mesh(path, model_folder):
    import smplx
    if not hasattr(smplx, "create"):
        module_path = getattr(smplx, "__file__", None) or getattr(smplx, "__path__", None)
        raise RuntimeError(
            "Imported module 'smplx' does not provide smplx.create(). "
            f"Imported location: {module_path}. "
            "This usually happens when the SMPL-X model folder is named 'smplx' "
            "inside the current project and shadows the Python package. "
            "Rename the model folder, for example to 'smplx_models', and pass that path "
            "to --smplx-model-root."
        )

    data = _load_smplx_params(path)
    transl = torch.from_numpy(data["transl"]).float()
    betas = torch.from_numpy(data["betas"]).float() if "betas" in data else None
    poses = torch.from_numpy(data["poses"]).float().reshape(-1, 55, 3)
    gender = data["meta"]["gender"]
    nframe = poses.shape[0]

    global_orient = poses[:, 0, :3]
    body_pose = poses[:, 1:22, :]
    jaw_pose = poses[:, 22:23, :]
    leye_pose = poses[:, 23:24, :]
    reye_pose = poses[:, 24:25, :]
    left_hand_pose = poses[:, 25:40, :]
    right_hand_pose = poses[:, 40:55, :]
    expression = torch.zeros(nframe, 10)

    num_betas = betas.shape[1] if betas is not None else 10
    model = smplx.create(
        model_folder,
        model_type="smplx",
        gender=gender,
        use_face_contour=True,
        use_pca=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        ext="npz",
    )

    kwargs = dict(
        transl=transl,
        global_orient=global_orient,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        return_verts=True,
    )
    if betas is not None:
        kwargs["betas"] = betas
    output = model(**kwargs)
    return output.vertices.detach().cpu().numpy().squeeze(), model.faces


def detect_collision(mesh_fn, max_collisions, device, verbose=True):
    import trimesh
    from mesh_intersection.bvh_search_tree import BVH

    if isinstance(mesh_fn, str):
        input_mesh = trimesh.load(mesh_fn)
        if isinstance(input_mesh, trimesh.Scene):
            input_mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in input_mesh.geometry.values()
            ])
    elif isinstance(mesh_fn, trimesh.base.Trimesh):
        input_mesh = mesh_fn
    else:
        raise TypeError(f"Unsupported mesh input: {type(mesh_fn)}")

    if verbose:
        print("Number of triangles = ", input_mesh.faces.shape[0])

    vertices = torch.tensor(input_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.faces.astype(np.int64), dtype=torch.long, device=device)
    triangles = vertices[faces].unsqueeze(dim=0)
    bvh = BVH(max_collisions=max_collisions)

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    outputs = bvh(triangles)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    if verbose:
        print(f"Elapsed time: {(time.time() - start) * 1000} ms")
    outputs = outputs.detach().cpu().numpy().squeeze()
    collisions = outputs[outputs[:, 0] >= 0, :]
    if verbose:
        print("Number of collisions = ", collisions.shape[0])
        print("Percentage of collisions (%)", collisions.shape[0] / float(triangles.shape[1]) * 100)
    return collisions, input_mesh


def compute_contact_frequency(root, model_folder, max_collisions=20, device="cuda"):
    import trimesh

    device = torch.device(device)
    total_frames = 0
    collision_frames = 0

    for follower_name in sorted(os.listdir(root)):
        if not follower_name.endswith("_00.npy"):
            continue
        leader_name = follower_name.replace("_00.npy", "_01.npy")
        follower_path = os.path.join(root, follower_name)
        leader_path = os.path.join(root, leader_name)
        if os.path.isdir(follower_path):
            continue
        if not os.path.exists(leader_path):
            raise RuntimeError(f"Missing leader SMPL-X file for CF: {leader_path}")

        vertices1, faces1 = get_smplx_mesh(follower_path, model_folder)
        vertices2, faces2 = get_smplx_mesh(leader_path, model_folder)
        normals1 = compute_vertex_normals_efficient(vertices1, faces1.copy()).astype(float)
        normals2 = compute_vertex_normals_efficient(vertices2, faces2.copy()).astype(float)
        vertices1 = inflate_mesh_efficient(vertices1, normals1)
        vertices2 = inflate_mesh_efficient(vertices2, normals2)

        nframe = min(vertices1.shape[0], vertices2.shape[0])
        total_frames += nframe
        sample_collision_frames = []

        for frame_idx in tqdm(range(nframe), desc=f"CF {follower_name}", leave=False):
            input_mesh1 = trimesh.Trimesh(vertices1[frame_idx, :], faces1)
            input_mesh2 = trimesh.Trimesh(vertices2[frame_idx, :], faces2)
            col1, _ = detect_collision(input_mesh1, max_collisions, device, verbose=False)
            col2, _ = detect_collision(input_mesh2, max_collisions, device, verbose=False)

            input_mesh = trimesh.util.concatenate([input_mesh1, input_mesh2])
            col12, input_mesh = detect_collision(input_mesh, max_collisions, device, verbose=False)
            col2 += input_mesh1.faces.shape[0]
            self_collisions = np.concatenate([col1, col2])

            self_col = numpy2set(self_collisions)
            all_col = numpy2set(col12)
            collisions = np.array(list(all_col - self_col))
            n_collisions = len(collisions)
            if n_collisions > 0:
                sample_collision_frames.append(frame_idx)

        collision_frames += len(sample_collision_frames)

    if total_frames == 0:
        raise RuntimeError(f"No *_00.npy/*_01.npy SMPL-X pairs found in {root}")
    return float(collision_frames / total_frames * 100.0)
