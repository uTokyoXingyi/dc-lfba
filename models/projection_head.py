# Nov. 19: projection head for contrastive learning
import torch.nn as nn

# VGG
# class ProjectionHead(nn.Module):
#     def __init__(self, in_dim: int, feature_dim: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, in_dim, bias=False),
#             nn.BatchNorm1d(in_dim), # is this necessary?
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             nn.Linear(in_dim, feature_dim, bias=True)
#         )
#     def forward(self, x):
#         return self.net(x)

# ResNet
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=False),
            nn.Linear(in_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.net(x)