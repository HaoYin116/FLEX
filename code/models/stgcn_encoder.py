import os
import torch
import torch.nn as nn
from .st_gcn.st_gcn import Model
import numpy as np
import pandas as pd
import random

class STGCNFeatureExtractor(nn.Module):
    def __init__(self):
        super(STGCNFeatureExtractor, self).__init__()

        # 固定权重路径：项目根目录下 /models/gcn_weight.pth
        weight_path = os.path.join(os.path.dirname(__file__), 'gcn_weight.pth')

        self.model = Model(
            in_channels=3,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'uniform'},
            edge_importance_weighting=True
        )
        #self.model.eval()  # 冻结权重
        # 加载预训练权重
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only= True)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        #self.model.eval()  # 冻结权重


    def forward(self, x):
        """
        x: Tensor of shape (N, 3, T, V, M)
        return: Tensor of shape (N, 256)
        """
        #with torch.no_grad():
        #print(x.shape)
        features = self.model.extract_feature(x)[1]  # (N, 256, T, V, M)
        feature_vector = features.mean(dim=[2, 3, 4])  # 平均池化 → (N, 256)
        return feature_vector

def remap21_to_25(data21):
    mapping_25 = {
        0:  None,  1:  None,  2:   1,  3:  0,
        4:  7,     5:  8,     6:   9,  7:  10,
        8:  3,     9:  4,     10:  5,  11: 6,
        12: 16,    13: 17,    14: 18,  15: None,
        16: 11,    17: 12,    18: 13,  19: None,
        20: 2,     21: 10,    22: 10,  23: 6,  24: 6
    }
    T, _, C = data21.shape
    data25 = np.zeros((T,25,C), dtype=data21.dtype)
    for i25, i21 in mapping_25.items():
        if i21 is not None:
            data25[:, i25, :] = data21[:, i21, :]
    # SpineBase
    spine_base = 0.5*(data21[:,11,:] + data21[:,16,:])
    data25[:,0,:] = spine_base
    # SpineMid
    data25[:,1,:] = 0.5*(spine_base + data21[:,2,:])

    data25[:,19,:] = 0.5*(data21[:,14,:]+data21[:,15,:])
    data25[:,15,:] = 0.5*(data21[:,19,:]+data21[:,20,:])
    # fingers 用 Rhand(6) / LWrist(9) 填充
    return data25

def normalize_ntu_skeleton(data, hip_index=0, spine_index=1, epsilon=1e-6):
    """
    输入数据格式: [N, T, V, C] = [样本数, 帧数, 关节数(25), 坐标(x,y,z)]
    输出格式: [N, T, V, C] 归一化后的数据
    """
    # 确保输入为float类型
    data = data.astype(np.float32)
    
    # --- 步骤1: 以髋关节为中心 ---
    hip_coords = data[:, :, hip_index, :]          # 提取髋关节坐标 [N, T, C]
    hip_coords = hip_coords[:, :, np.newaxis, :]   # 扩展维度 [N, T, 1, C]
    normalized_data = data - hip_coords            # 广播减法 [N, T, V, C]
    
    # --- 步骤2: 骨骼长度归一化 ---
    # 计算脊柱长度（髋关节到脊柱中心）
    spine_vector = normalized_data[:, :, spine_index, :]  # 脊柱中心坐标 [N, T, C]
    spine_length = np.linalg.norm(spine_vector, axis=-1, keepdims=True)  # [N, T, 1]
    
    # 缩放所有关节坐标
    normalized_data = normalized_data / (spine_length[:, :, np.newaxis, :] + epsilon)
    
    return normalized_data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed()
    #data = np.load(os.path.join(os.path.dirname(__name__),'x.npy'))  # 举例
    emg_path = '/data/YH/FLEX-AQA/FLEX-AQA3/Skeleton/Skeleton/A01/099/skeleton_points.csv'
    raw_data = pd.read_csv(emg_path, header=0).values  # shape: (103, 25, 3)
    # 转换为 ST-GCN 输入格式
    raw_21 = raw_data.reshape(-1, 21, 3)  # (103, 21, 3)
    raw_25 = remap21_to_25(raw_21)
    skeleton_data = torch.from_numpy(raw_25).float()
    skeleton_data = skeleton_data.unsqueeze(0)  # → (1, 103, 25, 3)
    skeleton_data = normalize_ntu_skeleton(skeleton_data.numpy(), 0, 1)  # (1, 103, 25, 3)
    skeleton_data = torch.from_numpy(skeleton_data).float()  # (1, 103, 25, 3)

    x = skeleton_data.permute(0, 3, 1, 2).unsqueeze(-1) # → (1, 3, 103, 25, 1)

    #x = torch.from_numpy(data).float()             # (T, 25, 3)
    #x = x.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(-1)  # → (1, 3, 103, 25, 1)

    # 加载模型
    #extractor = STGCNFeatureExtractor(os.path.join(os.path.dirname(__file__), "gcn_weight.pth"))
    extractor = STGCNFeatureExtractor()
    # 提取特征
    feature_vector = extractor(x)  # → shape: (1, 256)

    print("特征 shape:", feature_vector.shape)
    arr = feature_vector.detach().numpy()               # numpy array shape (1, 512)
    df = pd.DataFrame(arr)                     # DataFrame shape (1, 512)
    df.to_csv('gcn_feature1.csv', index=False, header=False)