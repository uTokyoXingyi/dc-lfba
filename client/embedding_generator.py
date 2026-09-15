# after stage 1, generate embeddings using labeled data
import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from models.encoder import build_vgg19bn_encoder
from models.encoder import ResNet18Encoder, ResNet50Encoder, build_vgg19bn_encoder
from common.utils import load_encoder_checkpoint # function to load trained encoder

# -------------------------------
# Main entry: compute embeddings
# -------------------------------
def compute_embeddings(cfg: dict, device: torch.device):
    # encoder, _ = build_vgg19bn_encoder(pretrained=False) # VGG
    encoder = ResNet18Encoder(pretrained=False) # ResNet
    # encoder = ResNet50Encoder(pretrained=False)

    # load_encoder_checkpoint(cfg["encoder"]["ckpt_path"],encoder)
    
    encoder.to(device)
    encoder.eval()
    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    # print(encoder[0][0].weight[0,0,0,0])
    # load dataset
    dataset, dataloader = load_labeled_dataset(cfg["data"]["root"], cfg["data"]["image_size"], cfg["data"]["batch_size"], cfg["data"]["num_workers"])
    class_names = dataset.classes
    
    # f1 = generate_embeddings(encoder, dataloader, device)
    # torch.save(f1.cpu(), "checkpoints/compare/f1.pt")
    print(class_names)
    # # print(dataset.classes)
    print(dataset.class_to_idx)

    embeddings, labels = generate_embeddings(encoder, dataloader, device)
    # generate_embeddings(encoder, dataloader, device)
    
    save_embeddings(embeddings, labels, class_names, cfg["embeddings"]["output_path"])
    print("Embedding saved")
    print(embeddings.shape[1])
# -------------------------------
# load labeled data
# reuse torchvision.datasets
# -------------------------------
def load_labeled_dataset(data_root, image_size, batch_size, num_workers):
    """
    Directory structure:
        data_root/
            class_0/
            class_1/
            ...
    Returns:
        dataset, dataloader
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = datasets.ImageFolder(
        root=str(data_root),
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dataset, dataloader

# -------------------------------
# generate embeddings
# -------------------------------
@torch.no_grad()
def generate_embeddings(encoder, dataloader, device):
    all_embeddings = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        # print(images.shape)
        # print(labels.shape)
        z = encoder(images)   # (B, D)
        z = F.normalize(z, dim=1) # normalization
        print("f1 stats:", z.mean(), z.std(), z.abs().max())
        all_embeddings.append(z.cpu()) # [z1, z2, z3, ...], list of tensors
        all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0) # make the list of tensors a single tensor
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels
    # return z

# -------------------------------
# save embeddings
# -------------------------------
def save_embeddings(embeddings, labels, class_names, save_dir: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "class_names": class_names,
        },
        save_dir / "client_embeddings.pt"
    )