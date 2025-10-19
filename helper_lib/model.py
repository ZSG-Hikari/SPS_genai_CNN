import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                     # 32x16x16
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                     # 64x8x8
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))          # 128x1x1
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x): return self.head(self.features(x))

def get_model(model_name="CNN", num_classes=10):
    name = (model_name or "").upper()
    if name in ("CNN","SMALLCNN"):
        return SmallCNN(num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")
