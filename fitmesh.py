import sys
import os

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_edge_loss,
    mesh_normal_consistency
)
from pytorch3d.renderer import (
    TexturesUV,
    look_at_view_transform,
    MeshRenderer,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    FoVPerspectiveCameras
)

torch.autograd.set_detect_anomaly(True)

def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")

def visualize_prediction(predicted_mesh, renderer, target_image, title='', silhouette=False, savename=""):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    if savename != "":
        plt.savefig(f"png/{savename}")
    else:
        plt.show()
    plt.close()

class Scene:
    def __init__(self, mesh, cameras, lights, renderer, device, target=True):
        self.mesh = mesh
        self.cameras = cameras
        self.lights = lights
        self.renderer = renderer
        self._rendered = None
        if target:
            imgs = self.render_all()
            self._rendered = imgs.detach().to(device)
    
    def render_all(self):
        if self._rendered is not None:
            return self._rendered
        rgbs = list()
        for k in range(len(self.cameras)):
            rgbs.append(self.renderer(self.mesh, cameras=self.cameras[[k]], lights=self.lights))
        return torch.cat(rgbs, dim=0)
    
    def update_mesh(self, mesh):
        self.mesh = mesh

    @property
    def images(self):
        return self.render_all()
    
    @property
    def target_rgbs(self):
        return self.images[..., :3]
    
    @property
    def target_sils(self):
        return self.images[..., 3]
    
    def diff(self, scene_rgb, scene_sil, camera_batch_size=2):
        weight_rgb = 1.0
        weight_sil = 1.0
        weight_edge = 1.0
        weight_normal = 0.01
        weight_laplacian = 1.0

        loss_edge = mesh_edge_loss(self.mesh)
        loss_normal = mesh_normal_consistency(self.mesh)
        loss_laplacian = mesh_laplacian_smoothing(self.mesh, method="uniform")
        loss_sil = 0
        loss_rgb = 0

        for j in np.random.permutation(len(self.cameras)).tolist()[:camera_batch_size]:
            images = self.renderer(self.mesh, cameras=self.cameras[[j]], lights=self.lights)
            rgb = images[..., :3]
            sil = images[..., 3]
            loss_sil += ((sil - scene_sil.target_sils[j]) ** 2).mean()/camera_batch_size
            loss_rgb += ((rgb - scene_rgb.target_rgbs[j]) ** 2).mean()/camera_batch_size

        loss = weight_rgb*loss_rgb + weight_sil*loss_sil + weight_edge*loss_edge + weight_normal*loss_normal + weight_laplacian*loss_laplacian
        
        return loss
    
def train(
        target_scene_rgb,
        target_scene_sil,
        src_mesh, 
        renderer, 
        target_cameras, 
        lights, 
        epochs, 
        device):

    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts, sphere_verts_rgb], lr=1e-2)

    new_src_mesh = src_mesh.offset_verts(deform_verts)
    opt_scene = Scene(new_src_mesh, target_cameras, lights, renderer, device, target=False)

    for i in range(0, epochs):
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)
        new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)

        opt_scene.update_mesh(new_src_mesh)

        sum_loss = opt_scene.diff(target_scene_rgb, target_scene_sil)

        print(f"Loss requires_grad: {sum_loss.requires_grad}")  # Should be True
        print(f"Loss grad_fn: {sum_loss.grad_fn}")  # Should not be None (e.g., <AddBackward0> or similar)

        print(f'epoch {i}/{epochs} : loss = {sum_loss}')
        
        sum_loss.backward()

        print(f"deform_verts grad norm: {deform_verts.grad.norm().item() if deform_verts.grad is not None else 'None'}")
        print(f"sphere_verts_rgb grad norm: {sphere_verts_rgb.grad.norm().item() if sphere_verts_rgb.grad is not None else 'None'}")

        visualize_prediction(new_src_mesh, renderer, target_scene_rgb.target_rgbs[1], f'epoch {i}/{epochs}', False, f"frame{i}.png")

        optimizer.step()
    return new_src_mesh

def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    return mesh

def load_and_normalize_mesh(path, device):
    return normalize_mesh(load_objs_as_meshes([path], device=device))

def init_scene(device, num_views, light_position):
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    lights = PointLights(device=device, location=[light_position])
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    r = R[1].reshape(1,3,3)
    t = T[1].reshape(1,3)
    default_camera = FoVPerspectiveCameras(device=device, R=r, T=t) 
    return R, T, cameras, default_camera, lights

def get_silhouette_renderer(default_camera):
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=128, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
    )

    # Silhouette renderer 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=default_camera, 
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )
    return silhouette_renderer

def get_textured_renderer(device, default_camera, lights):
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=128, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
        perspective_correct=False, 
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=default_camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(device=device, 
            cameras=default_camera,
            lights=lights)
    )
    return renderer_textured



def main():
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

    rgb_scene = Scene(mesh, cameras, lights, rgb_renderer, device)
    sil_scene = Scene(mesh, cameras, lights, silhouette_renderer, device)
    

    print('rendered target images')
    

    src_mesh = ico_sphere(4, device)

    new_src_mesh = train(
        rgb_scene,
        sil_scene,
        src_mesh,
        rgb_renderer,
        cameras,
        lights,
        epochs,
        device
    )

    target_rgb = rgb_scene.target_rgbs
    visualize_prediction(
        new_src_mesh,
        renderer=rgb_renderer,
        target_image=target_rgb[1],
        silhouette=False
    )

if __name__ == '__main__':
    main()