import torch
import open3d as o3d
import numpy as np


from dataclasses import dataclass, field
import torch
import threestudio
from math import *
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render 
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
from Fourierembed import Embedder,get_embedder
import numpy as np
import io  
from PIL import Image  
import open3d as o3d
# 假设这些是从你的代码库导入的函数
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image 
import torch.nn as nn

import configargparse
import json
from Camera_create import create_camera_objects

def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def generate_3d_point_cloud_text(prompt, output_file='output.ply'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config_shape(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 15.0

    # 打印提示
    print('Prompt:', prompt)

    # 采样潜在向量
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # 解码潜在网格
    pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

    # 提取坐标和颜色
    coords = pc.verts
    rgb = np.concatenate([pc.vertex_channels['R'][:, None], 
                         pc.vertex_channels['G'][:, None], 
                         pc.vertex_channels['B'][:, None]], axis=1)

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # 颜色值归一化到 [0, 1]

    # 保存点云
    o3d.io.write_point_cloud(output_file, point_cloud)

    return coords, rgb, 0.4



def on_before_optimizer_step(self, optimizer):

        with torch.no_grad():
            
            if self.true_global_step < 900: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold) 




def config_parser():
    parser = configargparse.ArgumentParser()
    # 添加必要的参数
    parser.add_argument('--text_encoder', type=str, required=True, help='Path to the text encoder model')
    parser.add_argument('--text_description', type=str,required=True,help= 'descripe what 3d obj you want')
    parser.add_argument('--sh_degree', type=int, default=2, help='Degree of spherical harmonics')
    parser.add_argument('--camera_pose_json_path', type=str, required=True, help='Path to the JSON file containing camera poses')
    parser.add_argument('--camera_net', type=str, required=True, help='Path to the camera network model')
    parser.add_argument('--diffusion_model_2d', type=str, required=True, help='Path to the 2D diffusion model')
    parser.add_argument('--cameras_extent', type=float, default=4.0, help='Extent for the cameras')
    parser.add_argument('--image2_3D_XM', type=str, default='transmitter', help='XM of image2_3D')
    parser.add_argument('--image2_3D_model', type=str, default='image300M', help='model of image2_3D')
    parser.add_argument('--image2_3D_md', type=str, default='diffusion', help='diffusion model of image2_3D')
    parser.add_argument('--use_GPU', type=bool, default=True, help='use_GPU is True if ues GPU')
    parser.add_argument('--is_trainging',type=bool,default=False,help="for judgment train or inference")
    parser.add_argument('--img_gt_path',type=str,help="if is_trainging, this is the ground truth of imgs")
    # 其他可能的参数...
    
    return parser.parse_args()


class Image_2_3D(nn.Module):
        #换到别的文件中
        def __init__(self, args):
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device('cuda' if args.use_GPU else 'cpu')
            # 加载模型
            # args.image2_3D_XM = 'transmitter'
            # args.image2_3D_model = 'image300M'
            # args.md = 'diffusion'
            self.xm = load_model(args.image2_3D_XM, device=self.device)
            self.model = load_model(args.image2_3D_model, device=self.device)  # 使用基于图像的模型
            #model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
            self.diffusion = diffusion_from_config_shape(load_config(args.image2_3D_md))
            self.batch_size = 1
            self.guidance_scale = 3.0  # 你可以根据需要调整这个值
            

        def forward(self, image):
        # 采样潜在向量
            latents = sample_latents(
                batch_size=self.batch_size,
                model=self.model,
                diffusion=self.diffusion,
                guidance_scale=self.guidance_scale,
                model_kwargs=dict(images=[image] * self.batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

            # 解码潜在网格
            pc = decode_latent_mesh(self.xm, latents[0]).tri_mesh()

            # 提取坐标和颜色
            coords = pc.verts
            rgb = np.concatenate([pc.vertex_channels['R'][:, None], 
                                pc.vertex_channels['G'][:, None], 
                                pc.vertex_channels['B'][:, None]], axis=1)

            print(rgb[0:5])
            if np.all((rgb >= 0) & (rgb <= 1)):
            # 创建Open3D点云对象
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coords)
                point_cloud.colors = o3d.utility.Vector3dVector(rgb)  # 不需要再次归一化
            else:
                # 如果颜色数据不是归一化的，则归一化到 [0, 1]
                rgb_normalized = rgb.astype(np.float64) / 255.0
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coords)
                point_cloud.colors = o3d.utility.Vector3dVector(rgb_normalized)

            # 保存由shape模型得到的点云
            # o3d.io.write_point_cloud(output_file, point_cloud)

            return coords, rgb, 0.4



        
class My_model(nn.Module):
    
    def __init__(self, args):
        super(My_model, self).__init__()
        self.text_encoder = self.load_encoder_model(args.text_encoder_path)
        self.text_description = args.text_description
        self.sh_degree = args.sh_degree
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        self.renderer = render  
        self.camera_net = self.load_camera_net_model(args.camera_net_path)
        self.diffusion_model_2d = self.load_diffusion_model(args.diffusion_model_2d)
        self.cameras_extent = args.cameras_extent
        self.point_clouds_generator = Image_2_3D(args)
        self.cameras , self.camera_poses = create_camera_objects(args.camera_pose_json_path)
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 初始化其他需要的组件...
        if args.is_training:
                self.imgs_gt = load_image(args.img_gt_path)

    def load_encoder_model(self, path):
        """
            该函数用于加载文本编码器模型，并返回一个模型对象。
            该模型的输入是文本，输出是文本的特征向量
        """
        return torch.load(path)
    def load_camera_net_model(self, path):
        """
            该函数用于加载相机视角计算网络模型，并返回一个模型对象。
            该模型的输入是k组图像，输出是预测的k个相机角度
        """
        return load_model(path)

    def load_diffusion_model(self, path):
        """
            该函数用于加载2D图像的扩散模型，并返回一个模型对象。
            该模型的输入是粗粒度的2D图像，输出是细粒度的图像。
        """
        return load_model(path)
    def add_points(self,coords,rgb):#增加扰动
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)
        num_points = 1000000  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))
        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)
        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))
        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb


    def pcb(self,image): #通过shape模型初始化点云

        coords,rgb,scale = self.generate_3d_point_cloud_from_image(image)
        
        bound= self.radius*scale

        all_coords,all_rgb = self.add_points(coords,rgb)
        

        pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((all_coords.shape[0], 3)))

        return pcd
    

    def forward(self, text_embedding, camera, renderbackground):
        
        """
        前向过程：[计算一组(k)相机参数下]
        1. 输入：文本嵌入、shape = [batch,token,feature]
                相机姿态、 shape = ??
                图像、     shape = [batch,c,H,W]
                背景颜色（可选）
        2. 输出：给定角度下预测的图像 shape = [batch,c,H,W]
                相机姿态  shape = ??  
                细粒度图像 shape = [batch,c,H,W]

        """
        ####################################################################################
        #          此处需要从camera对象中提取出对应的需要编码的信息                            #
        pose_embedding = self.Fourierembedding_of_cam(text_embedding.shape[1],camera)      #       
        ####################################################################################
        #暂时定的pose中包括  'c2w'、'Fovx'、,'Fovy'、,'height'、'width',
        mixed_embedding = text_embedding + pose_embedding  # 可以选择更复杂的融合方法

        mixed_feature = self.text_encoder(mixed_embedding)
        
        render_pkg = self.renderer(camera_poses, self.gaussian, self.pipe, renderbackground)
        coarse_images, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        self.viewspace_point_list.append(viewspace_point_tensor)
        
        # 添加噪声
        noise = torch.randn_like(coarse_images)
        coarse_img_noise = coarse_images + noise
        
        # 通过2D扩散模型去噪
        fine_img = self.diffusion_model_2d.denoise(coarse_img_noise, mixed_feature)
        
        # 预测相机姿态
        predicted_pose = self.camera_net.predict(fine_img)
        return fine_img, predicted_pose
    def Fourierembedding_of_cam(self,multires,camera):
        Fourierembedding, out_dim= get_embedder(log(multires))
        return Fourierembedding(camera)

def train(self, num_epochs=100,renderbackground = None):
    self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    if renderbackground is None:
        renderbackground = self.background_tensor
    # 文本编码
    text_embedding = self.text_encoder.encode(self.text_description)
    
    # 使用正面图像初始化3D点云并构建高斯模型
    front_image = self.imgs_gt[0]
    optimizer = OptimizationParams(self.parser)

    point_cloud = self.pcb(front_image)
    self.gaussian_model.create_from_pcb(point_cloud,self.cameras_extent)
    self.pipe = PipelineParams(self.parser)
    self.gaussian_model.training_setup(optimizer)
    

    
    criterion_img = nn.MSELoss()
    # 假设有一个适当的相机姿态损失函数
    criterion_pose = Camera_Loss_Function()

    criterion_vae = VaeLoss()
    for epoch in range(num_epochs):
        #？？？？这一部分不知道什么作用
        # self.gaussian_model.update_learning_rate(self.true_global_step)
        
        # if self.true_global_step > 500:
        #     self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        # self.gaussian_model.update_learning_rate(self.true_global_step)

        images = []
        losses = 0
        for i, (image_gt, pose) in enumerate(zip(self.imgs_gt, self.camera_poses)):
            loss = 0
            # 姿态编码
            fine_img, predicted_pose = self(text_embedding, pose, self.cameras[i], renderbackground)               
            # 计算损失
            img_loss = criterion_img(fine_img, image_gt)
            pose_loss = criterion_pose(predicted_pose, pose)
            vae_loss = criterion_vae()
            loss = img_loss + pose_loss + vae_loss  # 可以调整权重
            
            losses += loss
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")
    


# 使用示例
if __name__ == "__main__":
    train()