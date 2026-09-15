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
from models.encoder import ResNet18Encoder #ResNet50Encoder, build_vgg19bn_encoder
from models.projection_head import ProjectionHead
from models.BYOLPredictor import BYOLPredictor
from models.SimSiamPredictor import SimSiamPredictor
from client_V0.datasets import ContrastiveDataset

# ============================================================
# Main entry point
# ============================================================
def train_encoder(cfg: Dict[str, Any], device) -> torch.nn.Module:
    name = cfg.get("method", {}).get("name", "simclr").lower()
    if name == "simclr":
        print("current is SimCLR")
        return train_simclr(cfg, device)
    elif name == "moco":
        print("current is MoCo")
        return train_moco(cfg, device)
    elif name == "byol":
        print("current is BYOL")
        return train_byol(cfg, device)
    elif name == "simsiam":
        print("current is SimSiam")
        return train_simsiam(cfg, device)
    else:
        raise ValueError(f"Unknown method: {name}")
    
# ============================================================
# Implement of SimCLR
# ============================================================
def train_simclr(cfg: Dict[str, Any], device) -> torch.nn.Module:
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
    ds = ContrastiveDataset(root=cfg["data"]["root"], image_size=cfg["data"]["image_size"], method = "simclr")
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)

    # 3. Optimizer
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=cfg["training"]["learning_rate"],
        momentum = cfg["optimizer"]["momentum"],
        weight_decay=float(cfg["training"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )
    # init_csv_logger(cfg["checkpoint"]["ssl_csv_path"])
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
        # print(f"Epoch {epoch}: lr = {current_lr:.6f}")
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

        # with torch.no_grad():
        #     z_i_norm = z_i.norm(dim=1).mean().item() # z_i = [batch_size, embedding_dim], norm(dim=1) computes the L2 norm
        #     # zj_norm = zj.norm(dim=1).mean().item()
        
        # with torch.no_grad():
        #     zi = F.normalize(z_i, dim=1)
        #     zj = F.normalize(z_j, dim=1)
        #     sim_matrix = zi @ zj.T
        #     mask = torch.eye(zi.size(0), device=zi.device).bool()
        #     pos_sim = sim_matrix.diag().mean().item()
        #     neg_sim = sim_matrix[~mask].mean().item()
        #     # pos_sim = F.cosine_similarity(zi, zj, dim=1).mean().item()
        #     # zj_neg = zj[torch.randperm(zj.size(0))] # not actually find the negative, just random the images in a batch, so they don't paired.
        #     # neg_sim = F.cosine_similarity(zi, zj_neg, dim=1).mean().item() 
        
        loss = contrastive_loss(z_i, z_j,temperature=cfg["training"]["temperature"])
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        # if step % cfg["logging"]["log_interval"] == 0: # log every log_interval, it could be set as mini-batch size
        #     print(
        #         f"[SSL] Epoch [{epoch}/{cfg['training']['epochs']}] " # which epoch
        #         f"Step [{step}/{len(dataloader)}] " # which step in mini batch
        #         f"Loss: {loss.item():.4f}" # print current loss.
        #         f"||z||: {z_i_norm:.2f} | "
        #         f"pos_sim: {pos_sim:.3f} | "
        #         f"neg_sim: {neg_sim:.3f}"
        #     )
        #     log_to_csv(
        #         csv_path=cfg["checkpoint"]["ssl_csv_path"],
        #         epoch=epoch,
        #         step=step,
        #         loss=loss.item(),
        #         emb_norm=z_i_norm,
        #         pos_sim=pos_sim,
        #         neg_sim=neg_sim
        #     )

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


# ============================================================
# Implement of MoCo
# ============================================================
@torch.no_grad()
def _momentum_update(model_q: nn.Module, model_k: nn.Module, m: float):
    """EMA update for key encoder/projection: k = m*k + (1-m)*q"""
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=(1.0 - m))

@torch.no_grad()
def _dequeue_and_enqueue(queue: torch.Tensor, queue_ptr: torch.Tensor, keys: torch.Tensor):
    """
    queue: [D, K]
    keys:  [B, D] (already normalized)
    queue_ptr: scalar int64 tensor
    """
    K = queue.size(1)
    B = keys.size(0)

    ptr = int(queue_ptr.item())
    # If B > K, keep last K keys (rare for your batch sizes)
    if B >= K:
        queue[:, :] = keys[-K:].T
        queue_ptr[0] = 0
        return

    # wrap-around
    end = ptr + B
    if end <= K:
        queue[:, ptr:end] = keys.T
    else:
        first = K - ptr
        queue[:, ptr:K] = keys[:first].T
        queue[:, 0:(B - first)] = keys[first:].T

    queue_ptr[0] = (ptr + B) % K

def moco_contrastive_loss(q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float):
    """
    q: [B, D] normalized query embeddings
    k: [B, D] normalized key embeddings (positive)
    queue: [D, K] normalized negative keys
    """
    # positive logits: [B, 1]
    l_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(1)
    # negative logits: [B, K]
    l_neg = torch.einsum("bd,dk->bk", [q, queue])

    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive is index 0
    return F.cross_entropy(logits, labels), l_pos.mean().item(), l_neg.mean().item()

def train_moco(cfg: Dict[str, Any], device) -> torch.nn.Module:
    """
    MoCo-style SSL training (single GPU):
    - encoder_q + proj_q are trained by SGD
    - encoder_k + proj_k are updated by momentum (EMA)
    - negatives come from a queue of previous keys
    """
    # ---- hyperparams (with safe defaults) ----
    moco_cfg = cfg.get("moco", {})
    m = float(moco_cfg.get("momentum", 0.99))          # EMA momentum
    K = int(moco_cfg.get("queue_size", 512))          # queue size
    proj_dim = int(cfg["model"]["projection_dim"])
    T = float(cfg["training"]["temperature"])
    # T = 0.2

    # ---- models ----
    encoder_q = ResNet18Encoder(pretrained=False)
    proj_q = ProjectionHead(encoder_q.feat_dim, proj_dim)

    encoder_k = ResNet18Encoder(pretrained=False)
    proj_k = ProjectionHead(encoder_k.feat_dim, proj_dim)

    encoder_q.to(device); proj_q.to(device)
    encoder_k.to(device); proj_k.to(device)

    encoder_k.load_state_dict(encoder_q.state_dict(), strict=True)
    proj_k.load_state_dict(proj_q.state_dict(), strict=True)

    # key encoder not updated by gradient
    for p in encoder_k.parameters():
        p.requires_grad_(False)
    for p in proj_k.parameters():
        p.requires_grad_(False)
    
    # ---- queue buffers ----
    # queue: [D, K]
    queue = torch.randn(proj_dim, K, device=device)
    queue = F.normalize(queue, dim=0)
    queue_ptr = torch.zeros(1, dtype=torch.long, device=device)

    # ---- data ----
    ds = ContrastiveDataset(root=cfg["data"]["root"], image_size=cfg["data"]["image_size"], method = "moco")
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.SGD(
        list(encoder_q.parameters()) + list(proj_q.parameters()),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    init_csv_logger(cfg["checkpoint"]["ssl_csv_path"])

    for epoch in range(cfg["training"]["epochs"]):
        encoder_q.train()
        proj_q.train()
        # encoder_k/proj_k stay in eval (BN uses running stats). This is fine for single-GPU MoCo.
        encoder_k.eval()
        proj_k.eval()

        for step, (x_i, x_j) in enumerate(loader):
            x_i = x_i.to(device, non_blocking=True)  # query view
            x_j = x_j.to(device, non_blocking=True)  # key view

            # ---- forward: query ----
            q = proj_q(encoder_q(x_i))
            q = F.normalize(q, dim=1)

            # ---- forward: key (no grad) ----
            with torch.no_grad():
                # momentum update key enc/proj
                _momentum_update(encoder_q, encoder_k, m)
                _momentum_update(proj_q, proj_k, m)

                k = proj_k(encoder_k(x_j))
                k = F.normalize(k, dim=1)

            # ---- loss ----
            loss, pos_sim, neg_sim = moco_contrastive_loss(q, k, queue, temperature=T)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ---- update queue ----
            with torch.no_grad():
                _dequeue_and_enqueue(queue, queue_ptr, k)

            # ---- logging ----
            if step % cfg["logging"]["log_interval"] == 0:
                with torch.no_grad():
                    q_norm = q.norm(dim=1).mean().item()
                print(
                    f"[MoCo] Epoch [{epoch}/{cfg['training']['epochs']}] "
                    f"Step [{step}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"||q||: {q_norm:.2f} | "
                    f"pos_sim: {pos_sim:.3f} | "
                    f"neg_sim: {neg_sim:.3f}"
                )
                log_to_csv(
                    csv_path=cfg["checkpoint"]["ssl_csv_path"],
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    emb_norm=q_norm,
                    pos_sim=pos_sim,
                    neg_sim=neg_sim
                )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[MoCo] Epoch {epoch}: lr = {current_lr:.6f}")
    return encoder_q

# ============================================================
# Implement of BOYL
# ============================================================
@torch.no_grad()
def _momentum_update(model_online: nn.Module, model_target: nn.Module, m: float):
    """
    EMA update:
    target = m * target + (1 - m) * online
    """
    for p_o, p_t in zip(model_online.parameters(), model_target.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))
    for b_o, b_t in zip(model_online.buffers(), model_target.buffers()):
        b_t.copy_(b_o)

def byol_loss_fn(p: torch.Tensor, z: torch.Tensor):
    """
    BYOL loss:
    p: online prediction
    z: target projection (stop-grad)
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()

def train_byol(cfg: Dict[str, Any], device) -> torch.nn.Module:
    """
    Single-GPU BYOL training.
    Returns the online encoder.
    """
    byol_cfg = cfg.get("byol", {})
    ema_m = float(byol_cfg.get("ema_m", 0.99))
    proj_dim = int(cfg["model"]["projection_dim"])
    pred_dim = int(byol_cfg.get("pred_dim", proj_dim))

    # -------------------------
    # Online network
    # -------------------------
    encoder_online = ResNet18Encoder(pretrained=False).to(device)
    projector_online = ProjectionHead(encoder_online.feat_dim, proj_dim).to(device)
    predictor = BYOLPredictor(proj_dim, pred_dim, proj_dim).to(device)

    # -------------------------
    # Target network
    # -------------------------
    encoder_target = ResNet18Encoder(pretrained=False).to(device)
    projector_target = ProjectionHead(encoder_target.feat_dim, proj_dim).to(device)

    # initialize target from online
    encoder_target.load_state_dict(encoder_online.state_dict(), strict=True)
    projector_target.load_state_dict(projector_online.state_dict(), strict=True)

    # target network is not trained by gradient
    for p in encoder_target.parameters():
        p.requires_grad_(False)
    for p in projector_target.parameters():
        p.requires_grad_(False)

    # -------------------------
    # Dataset / Loader
    # -------------------------
    ds = ContrastiveDataset(
        root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        method = "byol"
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.SGD(
        list(encoder_online.parameters()) +
        list(projector_online.parameters()) +
        list(predictor.parameters()),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )
    init_csv_logger(cfg["checkpoint"]["ssl_csv_path"])

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(cfg["training"]["epochs"]):
        encoder_online.train()
        projector_online.train()
        predictor.train()

        encoder_target.eval()
        projector_target.eval()

        for step, (x_i, x_j) in enumerate(loader):
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            # ---- online branch ----
            z_i_online = projector_online(encoder_online(x_i))
            z_j_online = projector_online(encoder_online(x_j))

            p_i = predictor(z_i_online)
            p_j = predictor(z_j_online)

            # ---- target branch ----
            with torch.no_grad():
                z_i_target = projector_target(encoder_target(x_i))
                z_j_target = projector_target(encoder_target(x_j))

            # symmetric BYOL loss
            loss_i = byol_loss_fn(p_i, z_j_target.detach())
            loss_j = byol_loss_fn(p_j, z_i_target.detach())
            loss = 0.5 * (loss_i + loss_j)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # EMA update of target networks
            with torch.no_grad():
                _momentum_update(encoder_online, encoder_target, ema_m)
                _momentum_update(projector_online, projector_target, ema_m)

            # ---- logging ----
            if step % cfg["logging"]["log_interval"] == 0:
                with torch.no_grad():
                    emb_norm = z_i_online.norm(dim=1).mean().item()
                    cos_sim = F.cosine_similarity(
                        F.normalize(p_i, dim=1),
                        F.normalize(z_j_target, dim=1),
                        dim=1
                    ).mean().item()

                print(
                    f"[BYOL] Epoch [{epoch}/{cfg['training']['epochs']}] "
                    f"Step [{step}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"||z||: {emb_norm:.2f} | "
                    f"cos_sim: {cos_sim:.3f}"
                )

                log_to_csv(
                    csv_path=cfg["checkpoint"]["ssl_csv_path"],
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    emb_norm=emb_norm,
                    pos_sim=cos_sim,   # reuse this column as alignment metric
                    neg_sim=0.0        # BYOL has no negatives
                )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[BYOL] Epoch {epoch}: lr = {current_lr:.6f}")

    return encoder_online

# ============================================================
# Implement of SimSaim
# ============================================================
def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor):
    """
    SimSiam loss:
    p: prediction
    z: target projection (stop-gradient)
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def train_simsiam(cfg: Dict[str, Any], device) -> torch.nn.Module:
    """
    Single-GPU SimSiam training.
    Returns the trained encoder.
    """
    simsiam_cfg = cfg.get("simsiam", {})
    proj_dim = int(cfg["model"]["projection_dim"])
    pred_dim = int(simsiam_cfg.get("pred_dim", proj_dim))

    # -------------------------
    # Model components
    # -------------------------
    encoder = ResNet18Encoder(pretrained=False).to(device)
    projector = ProjectionHead(encoder.feat_dim, proj_dim).to(device)
    predictor = SimSiamPredictor(proj_dim, pred_dim, proj_dim).to(device)

    # -------------------------
    # Dataset / Loader
    # -------------------------
    ds = ContrastiveDataset(
        root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        method = "simsiam"
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )
    # -------------------------
    # Optimizer / Scheduler
    # -------------------------
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) +
        list(projector.parameters()) +
        list(predictor.parameters()),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )
    init_csv_logger(cfg["checkpoint"]["ssl_csv_path"])

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(cfg["training"]["epochs"]):
        encoder.train()
        projector.train()
        predictor.train()

        for step, (x_i, x_j) in enumerate(loader):
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            # ---- branch 1 ----
            z_i = projector(encoder(x_i))   # projection
            p_i = predictor(z_i)            # prediction

            # ---- branch 2 ----
            z_j = projector(encoder(x_j))
            p_j = predictor(z_j)

            # stop-gradient on opposite branch
            loss_i = negative_cosine_similarity(p_i, z_j.detach())
            loss_j = negative_cosine_similarity(p_j, z_i.detach())
            loss = 0.5 * (loss_i + loss_j)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ---- logging ----
            if step % cfg["logging"]["log_interval"] == 0:
                with torch.no_grad():
                    emb_norm = z_i.norm(dim=1).mean().item()
                    cos_sim = F.cosine_similarity(
                        F.normalize(p_i, dim=1),
                        F.normalize(z_j.detach(), dim=1),
                        dim=1
                    ).mean().item()

                print(
                    f"[SimSiam] Epoch [{epoch}/{cfg['training']['epochs']}] "
                    f"Step [{step}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"||z||: {emb_norm:.2f} | "
                    f"cos_sim: {cos_sim:.3f}"
                )

                log_to_csv(
                    csv_path=cfg["checkpoint"]["ssl_csv_path"],
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    emb_norm=emb_norm,
                    pos_sim=cos_sim,   # reuse as alignment metric
                    neg_sim=0.0        # SimSiam has no negatives
                )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[SimSiam] Epoch {epoch}: lr = {current_lr:.6f}")

    return encoder