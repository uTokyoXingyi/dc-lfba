import torch.nn as nn

class BYOLPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# class BYOLPredictor(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, in_dim),
#             nn.BatchNorm1d(in_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_dim, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)