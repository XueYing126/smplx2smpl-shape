import os.path as osp

import numpy as np
import torch

import pyrender
import trimesh

import smplx
from smplx.joint_names import Body

from tqdm.auto import tqdm, trange

gender = 'female'
betas = [1.6, -2.5, 0.4, 0.9, 0.9, 0.6, -0.1, -0.4, 0.5, -0.4]


model_folder='./transfer_data/body_models'
model_type="smplx"
num_betas = 10

model = smplx.create(
    model_folder,
    model_type=model_type,
    gender=gender,
    use_face_contour=False,
    num_betas=num_betas,
    num_expression_coeffs=10,
    use_pca=False,
    ext='npz',
)

betas = torch.tensor(betas).float()
betas = betas.unsqueeze(0)[:, : model.num_betas]


output = model(
    betas=betas,

    global_orient=torch.zeros([1, 3]),
    body_pose=torch.zeros([1, 21*3]),

    return_verts=True,
)

vertices = output.vertices.detach().cpu().numpy().squeeze()
joints = output.joints.detach().cpu().numpy().squeeze()

vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
# process=False to avoid creating a new mesh
tri_mesh = trimesh.Trimesh(
    vertices, model.faces, vertex_colors=vertex_colors, process=False
)

output_path = './transfer_data/meshes/amass_sample/001.obj'
tri_mesh.export(output_path)