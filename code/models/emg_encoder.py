import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import pandas as pd
import matplotlib.pyplot as plt

class EMG_Spectrogram_ResNet18_Encoder(nn.Module):
    def __init__(self, sr=200, image_size=224):
        super().__init__()
        self.sr = sr                  # 采样率 (Hz)
        self.image_size = image_size  # 频谱图输入 ResNet 的图像尺寸

        # 使用预训练 ResNet18 (去掉最后分类层)
        base_model = models.resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'resnet_weight.pth'), map_location='cpu')
        
        if 'conv1.weight' in state_dict and state_dict['conv1.weight'].shape[1] != 1:
            del state_dict['conv1.weight']  # 删除不兼容的 conv1 权重
        
        base_model.load_state_dict(state_dict, strict=False)

        self.resnet = nn.Sequential(*list(base_model.children())[:-1])  # 输出 shape: (N, 512, 1, 1)
        #self.bn1=nn.BatchNorm2d(1)

        # Mel 频谱图转换器
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=100,
            hop_length=50,
            n_mels=64,
            f_min=20,
            f_max=sr // 2
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def forward(self, x):
        """
        x: Tensor of shape (M, N), M: 时间点数, N: 通道数
        返回: (1, 512) 特征向量
        """
        M, N = x.shape
        #print(x.shape)

        features = []
        for i in range(N):
            waveform = x[:, i]  # shape: (M,)
            spec = self.mel_transform(waveform.unsqueeze(0))  # (1, mel, time)
            spec_db = self.amplitude_to_db(spec)  # 转为 dB

            # Resize 到 224x224
            spec_db = F.interpolate(spec_db.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)  # shape: (1, 1, 224, 224)
            min_val = spec_db.min()
            max_val = spec_db.max()
            spec_db = (spec_db - min_val) / (max_val - min_val + 1e-6)
            #spec_db=self.bn1(spec_db)
            #print('spec_db',spec_db.mean(),spec_db.max(),spec_db.min() ,spec_db.shape)

            # Repeat 到 3 通道以适配 ResNet
            # spec_img = spec_db.repeat(1, 3, 1, 1)  # (1, 3, 224, 224)
            
            feat = self.resnet(spec_db)  # → (1, 512, 1, 1)
            feat = feat.view(1, -1)  # → (1, 512)
            features.append(feat)
        out = feat
        #feats = torch.cat(features, dim=0)
        #out = feats.mean(dim=0, keepdim=True)
        return out  # 最终特征
    
if __name__ == "__main__":
    #feat = EMG_Spectrogram_ResNet18_Encoder()
    
    emg_path = '/data/YH/FLEX-AQA/FLEX-AQA3/EMG/A01/199/EMG.csv'
    
    raw_data = pd.read_csv(emg_path,header=None).values

    emg_data = torch.from_numpy(raw_data).float()
    
    min_vals = emg_data.min(dim=0, keepdim=True)[0]
    max_vals = emg_data.max(dim=0, keepdim=True)[0]
    denom = (max_vals - min_vals).clamp(min=1e-6)  # 防止除以 0
    emg_data = (emg_data - min_vals) / denom
    
    emg_data = emg_data[:,1]
    emg_length = 500

    #feature = feat(emg_data)
    sr = 200  # 采样率
    n_fft = 100
    hop_length = 50
    n_mels = 64
    f_min = 20
    f_max = sr // 2

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    )
    amplitude_to_db = T.AmplitudeToDB(top_db=80)

    # 3. 计算 Mel 频谱图
    spec = mel_transform(emg_data.unsqueeze(0))       # [1, n_mels, time_frames]
    spec_db = amplitude_to_db(spec).squeeze(0)         # [n_mels, time_frames]
    spec_db = F.interpolate(spec_db.unsqueeze(0).unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False)
    

    print('ghgjhgjhg',spec_db.mean(),spec_db.max(),spec_db.min() ,spec_db.shape) 


    #spec_img = spec_db.squeeze(0).repeat(3,1,1)  # [1, 224, 224]
    #img = spec_img.permute(1, 2, 0).cpu().numpy()
    spec_img = spec_db.repeat(1, 3, 1, 1) 
    resnet = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # [N, 512, 1, 1]

    # 2. 前向
    with torch.no_grad():
        feat_map = backbone(spec_img)               # [1, 512, 1, 1]
        feat_vec = feat_map.view(feat_map.size(0), -1)

    arr = feat_vec.cpu().numpy()               # numpy array shape (1, 512)
    df = pd.DataFrame(arr)                     # DataFrame shape (1, 512)
    df.to_csv('feature.csv', index=False, header=False)
    