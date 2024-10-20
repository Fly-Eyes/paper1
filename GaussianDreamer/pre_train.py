import torch
import open3d as o3d
import numpy as np

#################################
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
from my_model import config_parser
import numpy as np
import io  
from PIL import Image  
import open3d as o3d
###############################
from torch.utils.data import Dataset
from PIL import Image
from my_model import *
from script_data_process import *

"""
该模块用于进行预训练，主要作用
0.导入text_encoder模块，并调用其forward函数进行训练。
1.导入my_model模块，并调用其forward函数进行训练。
2.设计损失，并反向传播以更新参数。
3.初始化（选择）文本编码器，并使用其编码文本描述。
4.定义数据集
"""





class MyDataset(Dataset):
    def __init__(self, data_path, random_list=False):
        """
        初始化数据集类，加载数据，并对其进行处理。
        
        Args:
            data_path (str): 数据集的根路径。
            random_list (bool): 是否对帧进行随机组合。
        """
        # 使用 traverse_directories_only 函数处理数据
        self.data = traverse_directories_only(data_path, random_list)

    def __len__(self):
        """
        返回数据集的长度，即所有序列组合的总数。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回一个组合的数据帧。

        Args:
            idx (int): 数据集中的索引。

        Returns:
            list: k个frame。
        """
        return self.data[idx]

    def load_image(self, path):
        """
        读取图像数据，转换为Tensor或适合的格式。
        这里可以选择使用PIL、OpenCV等库加载图片，具体实现取决于图像格式。

        Args:
            path (str): 图片文件路径。

        Returns:
            Tensor: 图像数据的Tensor表示。
        """
        # 可以根据你的需求来实现图像读取的方式
        # 例如使用PIL读取并转换为Tensor：
        image = Image.open(path)
        return torch.tensor(image)

def Camera_Loss_Function(predicted_pose, target_pose):
    pass

def Vae_Loss():
    return 0
def load_text_encoder(path):
    """
        该函数用于加载文本编码器模型，并返回一个模型对象。
        该模型的输入是文本，输出是文本的特征向量
    """
    return torch.load(path)
def pre_train(dataloader,num_epochs=100,renderbackground = None, text_description = None):
    parser = config_parser()
    # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    text_encoder = self.load_text_encoder(parser.text_encoder_path)
    
    if renderbackground is None:
        renderbackground = My_model.background_tensor
    
    
    My_model = My_model(parser)

    for epoch in range(parser.num_epochs):
        for batch_idx, data in enumerate(dataloader):
    # 使用正面图像初始化3D点云并构建高斯模型    
            """
                front_image
                cameras_extent
                数据集里面的内容
            """
            """
                data：# 类型为list，表示数据集中的所有数据。
                data[i]：# 类型为dict，表示第i个数据的信息。
                data[i]['image']：# 类型为dict，表示当前帧的图片信息，包含路径'path'和宽高'size'。
                data[i]['mask']：# 类型为dict，表示当前帧的图像掩码信息，包含路径'path'等。
                data[i]['viewpoint']：# 类型为dict，表示当前帧的相机位姿，包含路径R'R',T'T',焦距'focal_length'和基准点'principal_point'等。

            实例：
                {
                    'sequence_name': '540_79043_153212',
                    'image': {
                        'path': 'apple/540_79043_153212/images/frame000007.jpg', 
                        'size': [900, 2000]
                    }, 
                    'mask': {
                        'path': 'apple/540_79043_153212/masks/frame000007.png', 
                        'mass': 1000000
                    }, 
                    'viewpoint': {
                        'R': [
                            [-0.9928926825523376, -0.11898525059223175, 0.0025822923053056], 
                            [0.11892586946487427, -0.9927613139152527, -0.016780570149421692], 
                            [0.0045602405443787575, -0.01635420322418213, 0.9998558759689331]
                        ], 
                        'T': [-0.16740168631076813, 1.307915449142456, 14.042159080505371], 
                        'focal_length': [3.9203171730041504, 3.9222772121429443], 
                        'principal_point': [-0.0, -0.0], 
                        'intrinsics_format': 'ndc_isotropic'
                    }
                }
        
            """
            front_image = data[0]["image"]
            ################################################
            # 初始点云更新之后如何继续放到模型里面去做自监督？需要讨论一下。
            optimizer = OptimizationParams(parser)
            point_cloud = My_model.pcb(front_image)
            My_model.gaussian_model.create_from_pcb(point_cloud,data[1:]["viewpoint"])
            My_model.pipe = PipelineParams(parser)
            My_model.gaussian_model.training_setup(optimizer)
            
            ######################################################
            text_description = data[1:]["text_description"]
            text_embedding = text_encoder(text_description)

            # 姿态编码
            fine_img, predicted_pose, coarse_images =  My_model(text_embedding,data[1:]["viewpoint"], renderbackground)             
            # 计算损失
            img_loss = nn.MSELoss(fine_img, data[1:]["image"])
            pose_loss = Camera_Loss_Function(predicted_pose, data[1:]["viewpoint"])
            vae_loss = Vae_Loss(coarse_images)
            loss = img_loss + pose_loss + vae_loss  # 可以调整权重
            losses += loss
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")
    

