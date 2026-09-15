# Centralized evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse # arguments container
from typing import Dict, Any # used for type hint
import yaml
from pathlib import Path

from models.encoder import ResNet18Encoder, ResNet50Encoder, build_vgg19bn_encoder
from models.classifier_head import ResNet_head, LinearClassifier, build_vgg_head_deep
from evaluation.dataloader import get_test_loader
from common.utils import load_encoder_checkpoint
from common.utils import load_head_checkpoint

# from experiments.exp_common import EncoderPlusHead, evaluate_model

# =========================
# Main evaluation workflow
# =========================
def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load vgg19bn model
    # encoder, feat_dim = build_vgg19bn_encoder(pretrained=False)
    # classifier_head = build_vgg_head_deep(in_dim=feat_dim, num_classes=cfg["head"]["num_classes"])

    # ResNet
    encoder = ResNet18Encoder(pretrained=False)
    # encoder = ResNet50Encoder(pretrained=False)
    classifier_head = ResNet_head(in_dim=encoder.feat_dim, num_classes=cfg["head"]["num_classes"]).to(device)

    # classifier_head = LinearClassifier(in_dim=encoder.feat_dim, num_classes=cfg["head"]["num_classes"]).to(device)

    load_encoder_checkpoint(cfg["encoder"]["ckpt_path"], encoder)
    load_head_checkpoint(cfg["head"]["ckpt_path"], classifier_head)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    classifier_head.to(device)
    classifier_head.eval()

    # load dataset
    test_dataset, test_loader = get_test_loader(cfg["test_data"]["root"],cfg["test_data"]["image_size"],cfg["test_data"]["batch_size"],cfg["test_data"]["number_workers"])

    class_index = test_dataset.class_to_idx # check the index assigned to each class

    # Feb 27
    # model = EncoderPlusHead(encoder, classifier_head)
    # model.to(device)
    # model.eval()
    # loss, s_acc, b_acc = evaluate_model(model, test_loader, device, 9)
    # print(f"Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

    loss, acc = evaluate(encoder, classifier_head, test_loader, device)
    print(f"[Test] Loss: {loss:.4f} | Acc: {acc:.4f}")
    print("Class mapping:", class_index)

# ==============================
# load parameters from eval.yaml
# ==============================
def load_configs(eval_config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration files for stage 2.
    """
    eval_cfg_path = Path(eval_config_path)
    if not eval_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {eval_cfg_path}")
    
    with open(eval_cfg_path,"r") as f:
        eval_cfg = yaml.safe_load(f)
    
    return eval_cfg

# =========================
# evaluation
# =========================
@torch.no_grad()
def evaluate(encoder, head, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0

    for x, y in dataloader: #x: [B,C,H,W]; y: [B,1]
        x = x.to(device)
        y = y.to(device)
        feats = encoder(x) # feature
        feats = F.normalize(feats, dim=1)
        print("feats stats:", feats.mean(), feats.std(), feats.abs().max())
        logits = head(feats) # logits
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1) # select class with largest probability as prediction

        total += y.size(0) # B. obtain the total smaples in test dataset
        correct += (preds == y).sum().item() #preds == y is [1,0,0,1,...]Bx1 .sum() .item() == number of correct samples
        total_loss += loss.item() * y.size(0) #y.size(0)=B; loss.item()=the average loss in a batch. total_loss is the loss over all samples

    return total_loss / total, correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client-side workflow controller")
    parser.add_argument("--eval-config", type=str, required=True, help="Path to evaluation config (stage 3)")
    args = parser.parse_args()

    cfg = load_configs(eval_config_path=args.eval_config)
    print(cfg["test_data"]["root"])

    main(cfg)