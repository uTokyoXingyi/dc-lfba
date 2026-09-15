import torch.nn as nn
from torchvision import models


class LinearClassifier(nn.Module):
    """
    Construct for build_vgg_head
    linear classifier for downstream task.

    """
    def __init__(self, in_dim: int = 512, num_classes: int = 9):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self,x):
        return self.fc(x)

# class build_vgg_head(nn.Module):
#     """
#     Construct for build_vgg_head
#     2-layer classifier for downstream task.

#     """
#     def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 1024):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=False),
#             nn.Dropout(p=0.3),
#             nn.Linear(hidden_dim, num_classes)
#         )
    
#     def forward(self,x):
#         return self.net(x)

class build_vgg_head_deep(nn.Module):
    """
    Construct for build_vgg_head
    Deep Head
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        # self.fc = nn.Linear(in_dim, num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )
    
    def forward(self,x):
        return self.net(x)

class ResNet_head(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=False),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)