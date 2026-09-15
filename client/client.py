# Luo Xingyi, Dec. 17,2025
# this is used to control the workflow at client side
# 1: initialize
# 2: train encoder (SSL)
# 3: save encoder
# 4: handshake: notify server READY
# 5: wait for ACK (optional)
# 6: compute embeddings
# 7: send embeddings to server (possibly in rounds)

import argparse # arguments container
from typing import Dict, Any # used for type hint
import torch
import yaml
from pathlib import Path
import numpy as np
import random

from client.ssl_trainer_SimCLR import train_encoder
from common.utils import save_encoder_checkpoint
from client.embedding_generator import compute_embeddings

def parse_args(): # my own function -- store the procedure for arguments container
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Client-side workflow controller")

    parser.add_argument("--ssl-config", type=str, required=True, help="Path to SSL training config (stage 1)")
    parser.add_argument("--emb-config", type=str, required=True, help="Path to head training config (stage 2)")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to run encoder training")

    return parser.parse_args()

def load_configs(ssl_config_path: str, emb_config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration files for stage 1 and stage 2.
    """
    ssl_cfg_path = Path(ssl_config_path)
    emb_cfg_path = Path(emb_config_path)
    if not ssl_cfg_path.exists() or not emb_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {ssl_cfg_path or emb_cfg_path}")
    
    with open(ssl_cfg_path,"r") as f:
        ssl_cfg = yaml.safe_load(f)
    with open(emb_cfg_path,"r") as f:
        emb_cfg = yaml.safe_load(f)
    
    cfg = {
        "ssl": ssl_cfg,
        "head_emb": emb_cfg,
    }
    return cfg

# =========================
# Set random seeds
# =========================
def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# =========================
# Stage Transition
# =========================
def notify_server_ready(cfg: Dict[str, Any]):
    """
    Send a READY signal to the server to trigger stage 2.
    """
    raise NotImplementedError

def wait_for_server_ack(cfg: Dict[str, Any]):
    """
    Optionally wait for server acknowledgment before proceeding.
    """
    raise NotImplementedError


# =========================
# Main workflow
# =========================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_configs(
        ssl_config_path=args.ssl_config,
        emb_config_path=args.emb_config
    ) # load the paramter for both ssl (stage1) and distributed depploy stage (stage2)
    # -------- Stage 1 --------
    print(cfg["ssl"]["training"]["batch_size"])
    print(cfg["ssl"]["training"]["epochs"])
    print(cfg["ssl"]["training"]["learning_rate"])
    # encoder = train_encoder(cfg["ssl"], device)
    # save_encoder_checkpoint(cfg["ssl"]["checkpoint"]["save_dir"],encoder,cfg["ssl"]["training"]["epochs"])
    
    # -------- Stage transition --------
    # notify_server_ready(cfg["head"])
    # wait_for_server_ack(cfg["head"])

    # -------- Stage 2 --------
    compute_embeddings(cfg["head_emb"], device)
    # send_embeddings(cfg["head"])

if __name__ == "__main__":
    set_seed(42)
    main()
