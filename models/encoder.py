import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch

## VGG MODEL
def build_vgg19bn_encoder(pretrained: bool = False):
    """
    Build VGG-19 (BN) encoder.
    Returns:
        encoder: nn.Module
        feature_dim: int (dimension of output feature)
    """
    vgg_model = models.vgg19_bn(pretrained=pretrained) # create vgg model
    # vgg_model = models.vgg19_bn(pretrained=True) # create vgg model
    encoder = nn.Sequential(
            vgg_model.features,
            vgg_model.avgpool,
            nn.Flatten()
        )
    feat_dim = 512 * 7 * 7 # 25088
    return encoder, feat_dim

## ResNet
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # resnet = models.resnet18(pretrained=pretrained) # this is abondoned in torvision
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)

        # keep everything except the final FC
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,      # NOT inplace
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,   # output: [B, 512, 1, 1]
        )

        self.feat_dim = 512

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # [B, 512]
        return x

# ResNet50 Encoder
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # Take everything except the final FC
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,

            resnet.avgpool  # (B, 2048, 1, 1)
        )

        self.feat_dim = 2048

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # (B, 2048)
        return x