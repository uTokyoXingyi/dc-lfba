from pathlib import Path

import torch
import torch, torch.nn as nn

# from models.encoder import build_vgg19bn_encoder
# from models.encoder import ResNet18Encoder

def save_encoder_checkpoint(
    save_dir,
    encoder,
    epoch=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder": encoder.state_dict(),
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
        save_path = save_dir / f"encoder_epoch_{epoch}.pth"
    else:
        save_path = save_dir / "encoder.pth"
    
    # save_path = save_dir / "encoder.pth"
    torch.save(checkpoint, save_path)


def load_encoder_checkpoint(ckpt_path, encoder, map_location="cpu"):
    encoder_path = Path(ckpt_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {encoder_path}")
    
    checkpoint = torch.load(encoder_path, map_location=map_location)
    
    if "encoder" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'encoder' weights")
    # encoder.load_state_dict(checkpoint["encoder"], strict=False)
    # the following codes are used to check whether model is loaded sucessfully.
    missing, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return encoder

def load_head_checkpoint(ckpt_path, classifier_head, map_location="cpu"):
    classifier_head_path = Path(ckpt_path)
    if not classifier_head_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {classifier_head_path}")
    
    checkpoint = torch.load(classifier_head_path, map_location=map_location)
    
    if "classifier_head" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'classifier_head' weights")
    classifier_head.load_state_dict(checkpoint["classifier_head"], strict=False)
    # the following codes are used to check whether head is loaded sucessfully.
    missing, unexpected = classifier_head.load_state_dict(checkpoint["classifier_head"], strict=False)
    print("Head Missing keys:", missing)
    print("Head Unexpected keys:", unexpected)
    return classifier_head