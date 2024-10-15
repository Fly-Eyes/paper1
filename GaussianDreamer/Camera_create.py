import json
import torch
import numpy as np

# 假设这些辅助函数已经在你的代码中定义好了
from gaussiansplatting.scene.cameras import My_Camera


def load_camera_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_camera_objects(camera_pose_json_path):
    data = load_camera_data(camera_pose_json_path)
    camera_angle_x = data.get('camera_angle_x', None)
    frames = data.get('frames', [])
    
    cameras = []
    camera_poses = []
    for frame in frames:
        c2w = np.array(frame['transform_matrix'])
        Fov = camera_angle_x  # 假设所有帧的垂直视场角相同
        height = 800  # 假设图像高度为800
        width = 800  # 假设图像宽度为800
        
        # 创建Camera对象
        camera = My_Camera(
            c2w=torch.tensor(c2w, dtype=torch.float32),
            fov=Fov,
            height=height,
            width=width,
            fov_type='fovx',
            trans=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            scale=1.0,
            data_device="cuda"
        )
        poses = {
            'c2w':c2w,
            'Fovx':camera.FoVx,
            'Fovy': camera.FoVy,
            'height': height,
            'width':width,
        }
        camera_poses.append(poses)
        # camera_pose = {
        #     'c2w': c2w,
            
        #     'FoVy': FoVy,
        #     'height': height,
        #     'width': width,
        # }
        # camera_poses.append(camera_pose)

        cameras.append(camera)
    
    return cameras,camera_poses

if __name__ == "__main__":
    # 替换为你的JSON文件路径
    file_path = './camera_poses/transforms.json'
    
    # 加载JSON数据
    
    # 创建相机对象列表
    cameras = create_camera_objects(file_path)
    
    # 打印第一个相机的信息
    if cameras:
        first_camera = cameras[0]
        print(f"R: {first_camera.R}")
        print(f"T: {first_camera.T}")
        print(f"FoVx: {first_camera.FoVx}")
        print(f"FoVy: {first_camera.FoVy}")
        print(f"Image Height: {first_camera.image_height}")
        print(f"Image Width: {first_camera.image_width}")
        print(f"World View Transform: {first_camera.world_view_transform}")
        print(f"Projection Matrix: {first_camera.projection_matrix}")
        print(f"Full Proj Transform: {first_camera.full_proj_transform}")
        print(f"Camera Center: {first_camera.camera_center}")
    else:
        print("No camera data found.")