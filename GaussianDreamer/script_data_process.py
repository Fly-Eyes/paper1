import os
import gzip
import json
import subprocess



def line_combine(ls_data, k):
    all_combinations = []
    
    # 遍历每一个序列
    for sequence in ls_data:
        # 计算帧数
        num_frames = len(sequence)
        
        # 将每个序列的帧按照 k 进行组合
        for i in range(0, num_frames, k):
            # 如果剩余帧数少于 k 帧，跳过（可以根据需求调整是否跳过）
            if i + k <= num_frames:
                combination = sequence[i:i+k]
                all_combinations.append(combination)

    # 打乱所有的组合
    random.shuffle(all_combinations)
    
    return all_combinations
def random_combine(ls_data, k):
    all_combinations = []
    
    # 遍历每一个序列
    for sequence in ls_data:
        # 计算帧数
        num_frames = len(sequence)
        first_frame = sequence[0]
        # 将每个序列的帧按照 k 进行组合
        for i in range(1, num_frames, k - 1):
            # 如果剩余帧数少于 k 帧，跳过（可以根据需求调整是否跳过）
            if i + k <= num_frames:
                combination = [first_frame] + sequence[i:i+k]
                all_combinations.append(combination)

    # 打乱所有的组合
    random.shuffle(all_combinations)
    
    return all_combinations

def select_data(data):
    del data['frame_number']
    del data['frame_timestamp']
    del data['depth']
    del data['meta']
    del data['camera_name']
    data['text_description'] = data['image']['path'].split('/')[0]
    return data
def traverse_directories_only(data_path,random_list = False):
    """
        data_path为数据集路径，只需给到数据集co3d_dataset文件夹出现的位置即可
        例如您的数据集位置：/opt/data/private/code/My_Paper01/co3d_dataset
        则data_path = '/opt/data/private/code/My_Paper01/co3d_dataset'
    """
    # 列出当前文件夹中的所有项目
    items = os.listdir(data_path)
    ls_data = []
    # 遍历并筛选出文件夹
    for item in items:
        item_path = os.path.join(data_path, item)  # 获取完整路径
        if os.path.isdir(item_path) and os.listdir(item_path):  # 检查是否为文件夹
            with gzip.open(os.path.join(item_path, 'frame_annotations.jgz'), 'rt', encoding='utf-8') as f:
                datas = json.load(f)  # it is a list
            dic_temp = {}
            for data in datas:
                dic_temp.setdefault(data['sequence_name'], []).append(select_data(data))
                dic_temp[-1]['image']['path']= os.path.join(data_path, dic_temp[-1]['image']['path'])
                dic_temp[-1]['depth']['path']= os.path.join(data_path, dic_temp[-1]['depth']['path'])
                dic_temp[-1]['depth']['mask_path']= os.path.join(data_path, dic_temp[-1]['depth']['mask_path'])
                dic_temp[-1]['mask']['path']= os.path.join(data_path, dic_temp[-1]['mask']['path'])
            for seq in dic_temp:
                ls_data.append(dic_temp[seq])#seq_name被存储在ls_data[i][0]['sequence_name']
     
    return random_combine(ls_data) if random_list else line_combine(ls_data, 3)
    """
        解释一下数据集ls_data的格式
        ls_data：|类型为list|表示数据集中的所有序列                                       
        ls_data[i]：|类型为list|表示一个序列，里面包含多个帧                               
        ls_data[i][j]:|类型为list| 表示一个字典，包含第i个序列的第j个帧的信息。
        ls_data[i][j]['image']:|类型为dict|表示当前帧的图片信息，包含路径'path'和宽高'size'。
        |已删除|ls_data[i][j]['depth']:|类型为dict|表示当前帧的深度图信息，包含路径'path'、缩放尺寸'scale_adjustment'和掩码'mask_path'。
        ls_data[i][j]['mask']:|类型为dict|表示当前帧的图像掩码信息，包含路径'path'等。
        ls_data[i][j]['viewpoint']:|类型为dict|表示当前帧的相机位姿，包含路径R'R',T'T',焦距'focal_length'和基准点'principal_point'等。
        实例：
        {'sequence_name': '540_79043_153212', 
        |已删除|'frame_number': 7, 
        |已删除|'frame_timestamp': 0.3970149253731343, 
        'image': {'path': 'apple/540_79043_153212/images/frame000007.jpg', 'size': [900, 2000]}, 
        |已删除|'depth': {'path': 'apple/540_79043_153212/depths/frame000007.jpg.geometric.png', 'scale_adjustment': 1.0, 
                  'mask_path': 'apple/540_79043_153212/depth_masks/frame000007.png'}, 
        'mask': {'path': 'apple/540_79043_153212/masks/frame000007.png', 'mass': 1000000}, 
        'viewpoint': {'R': [[-0.9928926825523376, -0.11898525059223175, 0.0025822923053056], 
                      [0.11892586946487427, -0.9927613139152527, -0.016780570149421692], 
                      [0.0045602405443787575, -0.01635420322418213, 0.9998558759689331]], 
                      'T': [-0.16740168631076813, 1.307915449142456, 14.042159080505371], 
                      'focal_length': [3.9203171730041504, 3.9222772121429443], 
                      'principal_point': [-0.0, -0.0], 'intrinsics_format': 'ndc_isotropic'},

        |已删除|'meta': {'frame_type': 'test_unseen', 'frame_splits': ['singlesequence_apple_test_0_unseen'], 'eval_batch_maps': []},
        |已删除|'camera_name': None
        }
    """
            


    