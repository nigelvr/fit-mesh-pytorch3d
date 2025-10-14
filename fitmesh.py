import sys
import os

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch

from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import TexturesVertex

from utils import (
    visualize_prediction,
    display_mesh,
    mesh_loss,
    load_and_normalize_mesh,
    init_scene,
    get_silhouette_renderer,
    get_textured_renderer
)

torch.autograd.set_detect_anomaly(True)

assert len(sys.argv) == 3, f'usage: {sys.argv[1]} <device> <epochs>'
os.makedirs('png', exist_ok=True)
device = torch.device(sys.argv[1])
epochs = int(sys.argv[2])

# Initialize the scene
mesh = load_and_normalize_mesh(os.path.join("./data", "cow/cow.obj"), device)
num_views = 20
R, T, cameras, default_camera, lights = init_scene(device, num_views=num_views, light_position=[0,0,-3])

# init renderers
rgb_renderer = get_textured_renderer(device, default_camera, lights)
silhouette_renderer = get_silhouette_renderer(default_camera)

print('rendered target images')
# target images
rgb_l, sil_l = list(), list()
for k in range(len(cameras)):
    rgb_l.append(rgb_renderer(mesh, cameras=cameras[[k]], lights=lights))
    sil_l.append(silhouette_renderer(mesh, cameras=cameras[[k]], lights=lights))
target_rgbs = torch.cat(rgb_l, dim=0)[...,:3]
target_sils = torch.cat(sil_l, dim=0)[...,3]

src_mesh = ico_sphere(4, device) # initial mesh to deform into the target shape

verts_shape = src_mesh.verts_packed().shape
deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
optimizer = torch.optim.Adam([deform_verts, sphere_verts_rgb], lr=1e-2)

new_src_mesh = src_mesh.offset_verts(deform_verts)
for i in range(0, epochs):
    optimizer.zero_grad()

    new_src_mesh = src_mesh.offset_verts(deform_verts)
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)

    sum_loss = mesh_loss(new_src_mesh, cameras, rgb_renderer, lights, target_sils, target_rgbs)

    print(f"Loss requires_grad: {sum_loss.requires_grad}")
    print(f"Loss grad_fn: {sum_loss.grad_fn}")  

    print(f'epoch {i}/{epochs} : loss = {sum_loss}')
    
    sum_loss.backward()

    print(f"deform_verts grad norm: {deform_verts.grad.norm().item() if deform_verts.grad is not None else 'None'}")
    print(f"sphere_verts_rgb grad norm: {sphere_verts_rgb.grad.norm().item() if sphere_verts_rgb.grad is not None else 'None'}")

    visualize_prediction(new_src_mesh, rgb_renderer, target_rgbs[1], f'epoch {i}/{epochs}', False, f"frame{i}.png")

    optimizer.step()

visualize_prediction(
    new_src_mesh,
    renderer=rgb_renderer,
    target_image=target_rgbs[1],
    silhouette=False
)
