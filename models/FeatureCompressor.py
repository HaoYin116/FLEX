import torch
import torch.nn as nn

class FeatureCompressor(nn.Module):
    def __init__(self, input_dim=1536, output_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc[0].bias)

    def forward(self, x):
        return self.fc(x)
