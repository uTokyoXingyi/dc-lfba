# used to control the flow at server side
# logic flow:
# 1. WAIT_READY
# 2. SEND_ACK
# 3. Receive EMBEDDINGS
# 4. TRAIN_HEAD
# 5. SAVE_HEAD

# import os
import torch
import argparse # arguments container
from typing import Dict, Any # used for type hint
import yaml
from pathlib import Path
import numpy as np
import random

from server.receiver import receive_embeddings
from server.head_trainer import train_classifier_head

# from experiments.exp_common import EncoderPlusHead, evaluate_model
# from models.encoder import ResNet18Encoder
# from evaluation.dataloader import get_test_loader

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Server] Waiting for READY signal...")
    # wait_for_ready()
    print("[Server] ACK sent to client.")
    # send_ack()

    print("[Server] Receiving embeddings...")
    data = receive_embeddings(cfg["embeddings"]["root"]) # get embeddings

    embeddings = data["embeddings"]
    labels = data["labels"] # [0,1,2,3,...,9] this is actually the class index
    class_names = data["class_names"]

    embedding_dim = embeddings.shape[1]
    num_classes = len(class_names)

    print(f"[Server] Samples: {embeddings.shape}")
    print(class_names)
    print(labels.shape)

    # print(f"[Server] Samples: {embeddings.shape[0]}")
    # print(f"[Server] Embedding dim: {embedding_dim}")
    # print(f"[Server] Num classes: {num_classes}")
    # print(class_names)
    # print(labels)

    print("[Server] Training classifier head...")
    classifier_head = train_classifier_head(
        cfg=cfg,
        embeddings=embeddings,
        labels=labels,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        device=device,
    )
    # Feb 2026
    # encoder = ResNet18Encoder(pretrained=False)

    # model = EncoderPlusHead(encoder, classifier_head)
    # model.to(device)
    # model.eval()
    # test_dataset, test_loader = get_test_loader("data/evaluation/test",224,64,8)
    # loss, s_acc, b_acc = evaluate_model(model, test_loader, device, 9)
    # print(f"Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

    # save classifier head
    save_dir = cfg["checkpoint"]["head_save_path"]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "classifier_head.pth"
    # os.makedirs(os.path.dirname(cfg["checkpoint"]["head_save_path"]), exist_ok=True)
    checkpoint = {
        "classifier_head": classifier_head.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"[Server] Head saved to {cfg["checkpoint"]["head_save_path"]}")
    print("[Server] Phase 2 completed.")

# load parameters from head_train.yaml
def load_configs(headtrain_config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration files for stage 2.
    """
    headtrain_cfg_path = Path(headtrain_config_path)
    if not headtrain_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {headtrain_cfg_path}")
    
    with open(headtrain_cfg_path,"r") as f:
        headtrain_cfg = yaml.safe_load(f)
    
    return headtrain_cfg

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Client-side workflow controller")
    parser.add_argument("--headtrain-config", type=str, required=True, help="Path to head training config (stage 2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_configs(headtrain_config_path=args.headtrain_config)
    # cfg = {
    #     "embedding_path": "outputs/embeddings/client_embeddings.pt",
    #     "head_save_path": "checkpoints/head/classifier_head.pt",
    # }
    print(cfg["embeddings"]["root"])
    main(cfg)