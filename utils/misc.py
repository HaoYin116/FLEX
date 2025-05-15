import numpy as np
from pydoc import locate


def import_class(name):
    return locate(name)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def denormalize(label, class_idx, upper = 100.0):
    ''' 
    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    '''
    label_ranges = {
        1: (40.99, 93.79),
        2: (47.68, 96.69),
        3: (44.8, 95.2),
        4: (45.71, 95.24),
        5: (34.59, 95.49),
        6: (33.96, 93.4),
        7: (33.73, 93.98),
        8: (37.4, 100.0),
        9: (41.98, 100.0),
        10: (36.64, 93.89),
        11: (40.82, 93.2),
        12: (38.78, 92.52),
        13: (48.3, 94.56),
        14: (30.22, 94.24),
        15: (22.16, 96.41),
        16: (29.85, 94.03),
        17: (43.31, 91.08),
        18: (47.77, 96.82),
        19: (45.4, 92.53),
        20: (44.89, 89.77)
    }

    label_range = label_ranges[class_idx]

    true_label = (label.float() / float(upper)) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label

def normalize(label, class_idx, upper = 100.0):
    '''
    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    '''
    label_ranges = {
        1: (40.99, 93.79),
        2: (47.68, 96.69),
        3: (44.8, 95.2),
        4: (45.71, 95.24),
        5: (34.59, 95.49),
        6: (33.96, 93.4),
        7: (33.73, 93.98),
        8: (37.4, 100.0),
        9: (41.98, 100.0),
        10: (36.64, 93.89),
        11: (40.82, 93.2),
        12: (38.78, 92.52),
        13: (48.3, 94.56),
        14: (30.22, 94.24),
        15: (22.16, 96.41),
        16: (29.85, 94.03),
        17: (43.31, 91.08),
        18: (47.77, 96.82),
        19: (45.4, 92.53),
        20: (44.89, 89.77)
    }
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0]) ) * float(upper) 
    return norm_label


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