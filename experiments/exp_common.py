import os, csv, random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# from client.ssl_trainer import train_encoder
from client.ssl_trainer import train_encoder

from server.head_trainer import train_classifier_head


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def append_csv(path: str, row: Dict, fieldnames: List[str]):
    """
    Append one row to a CSV file. Creates file + header if missing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) # obtain state of the file

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # fill missing keys with ""
        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)

# data transformaction
def default_transform(img_size: int):
    # define the transformation
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

# obtain dataset which read data from folder
def make_imagefolder(root: str, transform):
    return datasets.ImageFolder(root=str(root), transform=transform)

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

def build_nested_subsets_imagefolder(dataset: datasets.ImageFolder, N_list: List[int], seed: int) -> Dict[int, List[int]]:
    rng = np.random.RandomState(seed) #random number generator
    targets = np.array(dataset.targets) # provides direct access to a list of all the integer labels (targets) assigned to the images in the dataset
    classes = np.unique(targets) # exclude the repeated ones

    # create a mapping where each unique class label c is a key, and its value is an array of indices from the targets array
    # {c: [index 1, index 2, ...]}
    per_class = {c: np.where(targets == c)[0] for c in classes} 
    for c in classes:
        rng.shuffle(per_class[c]) # shuffle the indexes of these images

    # nested = {
    # 5: []
    # 10: []
    # 25: []
    # }
    nested = {}
    for N in N_list:
        idx = []
        for c in classes:
            # if len(per_class[c]) < N:
            #     raise ValueError(f"Class {c} has {len(per_class[c])} samples < N={N}") #for now, we keep the data set balanced. that is N for all classes
            # idx.extend(per_class[c][:N].tolist())

            # unbalanced dataset
            if len(per_class[c]) < N:
                idx.extend(per_class[c][:len(per_class[c])].tolist())
            else:
                idx.extend(per_class[c][:N].tolist())
        nested[N] = sorted(idx)
    return nested

# -------------------------
# Models
# -------------------------
class EncoderPlusHead(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder(x)
        feats = F.normalize(z, dim=1) # embeddings has normalization.
        return self.head(feats)


# -------------------------
# Embedding generation
# -------------------------
@torch.no_grad()
def generate_embeddings(encoder, dataloader, device):
    all_embeddings = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        z = encoder(images)   # (B, D)
        z = F.normalize(z, dim=1) # normalization 
        # print("f1 stats:", z.mean(), z.std(), z.abs().max())
        all_embeddings.append(z.cpu()) # [z1, z2, z3, ...], list of tensors
        all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0) # make the list of tensors a single tensor
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels

# -------------------------
# Training stubs you plug in
# -------------------------
def train_encoder_contrastive_wrapper(
     method: str,
     epochs_ssl: int,
     batch_size: int,
     num_workers: int,
     lr: float,
     weight_decay: float,
     temperature: float,
     momentum: float,
     dir_unlabeled: str,
     image_size: int,
     save_dir: str,
     save_interval: int,
     ssl_csv_path: str,
     projection_dim: int,
     log_interval: int,
     device,
):
    cfg = {
        "method": {
            "name": method, # {simclr, moco, byol, simsiam}
        },
        "training": {
            "epochs": epochs_ssl,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
        },
        "optimizer": {
            "momentum": momentum,
        },
        "data": {
            "root": dir_unlabeled,
            "image_size": image_size,
        },
        "checkpoint": {
            "save_dir": save_dir,
            "save_interval": save_interval,
            "ssl_csv_path": ssl_csv_path,
        },
        "model": {
            "projection_dim": projection_dim,
        },
        "logging": {
            "log_interval": log_interval,
        },
    }
    if method == "moco":
        cfg["moco"] = {
            "queue_size": 512, #4096
            "momentum": 0.99,
        }
    elif method == "byol":
        cfg["byol"] = {
            "ema_m": 0.99,
            "pred_dim": 256,
        }
    elif method == "simsiam":
        cfg["simsiam"] = {
            "pred_dim": 256,
        }

    return train_encoder(cfg, device)


def train_model_supervised( 
         num_classes: int,
         device: torch.device,
         epochs: int = 150,
         batch_size: int = 256,
         num_workers: int = 8,
         lr: float = 0.01,
         weight_decay: float = 1e-4,
         momentum: float = 0.9,
         use_cosine: bool = True,
         log_every: int = 10,
         pretrained: bool = False,
         dir_labeled_train: str = None,
         transform=None,
         dataset=None,
        ) -> nn.Module:
    if dataset is not None:
        ds_train = dataset
    else:
        if dir_labeled_train is None or transform is None:
            raise ValueError("Must provide either dataset OR (dir_labeled_train + transform)")
        ds_train = datasets.ImageFolder(root=dir_labeled_train, transform=transform)
    
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
     )
    if pretrained:
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    else:
        model = models.resnet18(weights=None)
    # model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
     )
    scheduler = None
    if use_cosine:
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # training loop
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        # pbar = tqdm(dl_train, desc=f"Sup Train {epoch}/{epochs}", leave=False)
        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            total += bs
            correct += (logits.argmax(dim=1) == y).sum().item()

            # pbar.set_postfix(loss=float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        # if log_every and (epoch % log_every == 0):
        #     avg_loss = running_loss / max(1, total)
        #     acc = correct / max(1, total)
        #     curr_lr = optimizer.param_groups[0]["lr"]
        #     print(f"[Sup][Epoch {epoch:3d}] loss={avg_loss:.4f} acc={acc:.4f} lr={curr_lr:.6f}")

    return model

# -------------------------
# Train head from embeddings
# -------------------------
def train_head_wrapper( embeddings,
                        labels,
                        embedding_dim: int,
                        num_classes,
                        device,
                        batch_size,
                        momentum: int,
                        epochs,
                        lr,
                    ) -> nn.Module:
    cfg = {
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "momentum": momentum,
        },
        "checkpoint": {
            "head_save_path": None,
            "train_csv_path": None,
        }
    }
    return train_classifier_head(
        cfg=cfg,
        embeddings=embeddings,
        labels=labels,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        device=device,
    )

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0) # number of images in one batch
        total_loss += loss.item() * bs
        total_n += bs # total samples
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    logits = torch.cat(all_logits, 0)
    y = torch.cat(all_y, 0)

    # standard accuracy
    s_acc = standard_accuracy(logits, y)
    b_acc = balanced_accuracy(logits, y, num_classes=num_classes)
    return total_loss / max(1, total_n), s_acc, b_acc

@torch.no_grad()
def standard_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def balanced_accuracy(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    accs = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            accs.append((pred[mask] == y[mask]).float().mean().item())
    return float(np.mean(accs)) if accs else 0.0