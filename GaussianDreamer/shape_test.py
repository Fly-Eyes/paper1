import torch
import open3d as o3d
import numpy as np

# 假设这些是从你的代码库导入的函数
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image 
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


def generate_3d_point_cloud_image(image_path, output_file='output.ply'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)  # 使用基于图像的模型
    #model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config_shape(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 3.0  # 你可以根据需要调整这个值

    # 加载图像
    image = load_image(image_path)

    # 打印图像路径
    print('Image:', image_path)

    # 采样潜在向量
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
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

    print(rgb[0:5])
    if np.all((rgb >= 0) & (rgb <= 1)):
    # 创建Open3D点云对象
        print("无需归一化")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)  # 不需要再次归一化
    else:
        # 如果颜色数据不是归一化的，则归一化到 [0, 1]
        rgb_normalized = rgb.astype(np.float64) / 255.0
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb_normalized)
    # # 创建Open3D点云对象
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(coords)
    # point_cloud.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # 颜色值归一化到 [0, 1]

    # 保存点云
    o3d.io.write_point_cloud(output_file, point_cloud)

    return coords, rgb, 0.4

import configargparse
def config_parser():
    parser = configargparse.ArgumentParser()
    # 添加必要的参数
    parser.add_argument('--text_encoder', type=str, required=True, help='Path to the text encoder model')
    parser.add_argument('--sh_degree', type=int, default=2, help='Degree of spherical harmonics')
    parser.add_argument('--camera_pose_json_path', type=str, required=True, help='Path to the JSON file containing camera poses')
    parser.add_argument('--camera_net', type=str, required=True, help='Path to the camera network model')
    parser.add_argument('--diffusion_model_2d', type=str, required=True, help='Path to the 2D diffusion model')
    parser.add_argument('--cameras_extent', type=float, default=4.0, help='Extent for the cameras')
    parser.add_argument('--image2_3D_XM', type=str, default='transmitter', help='XM of image2_3D')
    parser.add_argument('--image2_3D_model', type=str, default='image300M', help='model of image2_3D')
    parser.add_argument('--image2_3D_md', type=str, default='diffusion', help='diffusion model of image2_3D')
    parser.add_argument('--use_GPU', type=bool, default=True, help='use_GPU is True if ues GPU')
    # 其他可能的参数...
    
    return parser.parse_args()
# 使用示例
if __name__ == "__main__":
    prompt = "A lego excavator"
    text_output_file = "./shapeoutput/lego.ply"
    image_output_file = "./shapeoutput/chair.ply"
    image_path = "./images/chair.png"

    # coords, rgb, _ = generate_3d_point_cloud_image(image_path,image_output_file)
    from my_model import Image_2_3D
    args = config_parser()
    point_clouds_generator = Image_2_3D(args)
    image = load_image(image_path)
    coords ,rgb,scale = point_clouds_generator(image)
    print(f"Point cloud saved to {image_output_file}")