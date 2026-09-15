# Train an encoder using self-supervised contrastive learning.
# interface: train_ssl_encoder(cfg, device) -> torch.nn.Module
# what it includes:
# 1: Load unlabeled dataset
# 2: Build encoder + projection head
# 3: Define contrastive loss
# 4: Training loop
# 5: Return trained encoder

from typing import Dict, Any
import csv
from pathlib import Path
import numpy as np
import random

import torch
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from models.encoder import build_vgg19bn_encoder
from models.encoder import ResNet18Encoder, ResNet50Encoder, build_vgg19bn_encoder
from models.projection_head import ProjectionHead
from client.datasets import ContrastiveDataset

# ============================================================
# Main entry point
# ============================================================
def train_encoder(cfg: Dict[str, Any], device) -> torch.nn.Module:
    # 1. Models -- VGG
    # encoder, encoder_dim = build_vgg19bn_encoder(pretrained=False)
    # projection_head = ProjectionHead(encoder_dim, cfg["model"]["projection_dim"])
    
    # Models -- ResNet
    encoder = ResNet18Encoder(pretrained=False)
    # encoder = ResNet50Encoder(pretrained=True)
    
    projection_head = ProjectionHead(encoder.feat_dim, cfg["model"]["projection_dim"])

    encoder.to(device)
    projection_head.to(device)

    # 2. Dataset
    ds = ContrastiveDataset(root=cfg["data"]["root"], image_size=cfg["data"]["image_size"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)

    # 3. Optimizer
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=cfg["training"]["learning_rate"],
        momentum = cfg["optimizer"]["momentum"],
        weight_decay=float(cfg["training"]["weight_decay"])
    ) #Adam -> SGD
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )
    init_csv_logger(cfg["checkpoint"]["ssl_csv_path"])
    # 4. Training loop
    for epoch in range(cfg["training"]["epochs"]):
        encoder.train()
        projection_head.train()
        run_one_epoch(
            epoch=epoch,
            dataloader=loader,
            encoder=encoder,
            projection_head=projection_head,
            optimizer=optimizer,
            device=device,
            cfg=cfg
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: lr = {current_lr:.6f}")
    return encoder

# =============================
# Training steps in one epoch
# =============================
def run_one_epoch(epoch: int, dataloader: DataLoader, encoder: torch.nn.Module, projection_head: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer, device, cfg: Dict[str, Any]
                  ):
    # inner mini-batch loop
    for step, (x_i, x_j) in enumerate(dataloader):
        # x_i, x_j: two image tensor, they are two augmented versions of the same image tensors
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        # forward pass
        z_i = projection_head(encoder(x_i))
        z_j = projection_head(encoder(x_j))

        with torch.no_grad():
            z_i_norm = z_i.norm(dim=1).mean().item() # z_i = [batch_size, embedding_dim], norm(dim=1) computes the L2 norm
            # zj_norm = zj.norm(dim=1).mean().item()
        
        with torch.no_grad():
            zi = F.normalize(z_i, dim=1)
            zj = F.normalize(z_j, dim=1)
            sim_matrix = zi @ zj.T
            mask = torch.eye(zi.size(0), device=zi.device).bool()
            pos_sim = sim_matrix.diag().mean().item()
            neg_sim = sim_matrix[~mask].mean().item()
            # pos_sim = F.cosine_similarity(zi, zj, dim=1).mean().item()
            # zj_neg = zj[torch.randperm(zj.size(0))] # not actually find the negative, just random the images in a batch, so they don't paired.
            # neg_sim = F.cosine_similarity(zi, zj_neg, dim=1).mean().item() 
        
        loss = contrastive_loss(z_i, z_j,temperature=cfg["training"]["temperature"])
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        if step % cfg["logging"]["log_interval"] == 0: # log every log_interval, it could be set as mini-batch size
            print(
                f"[SSL] Epoch [{epoch}/{cfg['training']['epochs']}] " # which epoch
                f"Step [{step}/{len(dataloader)}] " # which step in mini batch
                f"Loss: {loss.item():.4f}" # print current loss.
                f"||z||: {z_i_norm:.2f} | "
                f"pos_sim: {pos_sim:.3f} | "
                f"neg_sim: {neg_sim:.3f}"
            )
            log_to_csv(
                csv_path=cfg["checkpoint"]["ssl_csv_path"],
                epoch=epoch,
                step=step,
                loss=loss.item(),
                emb_norm=z_i_norm,
                pos_sim=pos_sim,
                neg_sim=neg_sim
            )

# =================================================
# build loss contrastive function
# reference: A Simple Framework for ..
# ..Contrastive Learning of Visual Representations
# =================================================
def contrastive_loss(z_i, z_j, temperature=0.5):
    """
    Build contrastive loss function
    """
    batch_size = z_i.size(0)
    # normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.matmul(z, z.T) / temperature # the kth row is similariyt between x_i^k and all 2N
    # let (k,k)-self-similarity be -inf
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool) # mask is a 2N*2N matrix, (i,j)=True (i=j) or False(i!=j).
    sim.masked_fill(mask, float("-inf")) # let self-simi be -inf, the exp(-inf)=0 #(i,j) of sim will be assign "-inf" when (i,j) of mask is True.

    # Jan. 28, 2026
    # the first version is wrong !!!!!

    # targets = torch.cat([
    #     torch.arange(batch_size, 2 * batch_size),
    #     torch.arange(0, batch_size)
    # ]).to(z.device) #targets=[N,N+1,...,2N-1,0,1,2,3,...,N-1]; this is the index of the correct class for each sample.
    # loss = F.cross_entropy(sim, targets) # for each row i, loss = sim[i,targets(i)]/sum_j(i,j). for j=i, item = 0.

    # new version of loss calculation
    #positives: (i,i + batch_size) and (i + batch_size, i)
    pos = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)],dim=0)
    loss = -pos + torch.logsumexp(sim, dim = 1)
    return loss.mean()

# =================================================
# log the parameters during training
# arguments: scv_path
# =================================================
def init_csv_logger(csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "step",
            "loss",
            "emb_norm",
            "pos_sim",
            "neg_sim"
        ])
def log_to_csv(
    csv_path,
    epoch,
    step,
    loss,
    emb_norm,
    pos_sim,
    neg_sim
):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            step,
            loss,
            emb_norm,
            pos_sim,
            neg_sim
        ])

# =================================================
# set the set of worker seeds
# ensure Data order is same
# =================================================
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 # return the random seed for the current process; and change it to 32 bits
    np.random.seed(worker_seed)
    random.seed(worker_seed)