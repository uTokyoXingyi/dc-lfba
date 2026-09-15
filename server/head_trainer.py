from typing import Dict, Any
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from models.classifier_head import build_vgg_head
from models.classifier_head import ResNet_head, LinearClassifier, build_vgg_head_deep
from server.EmbDataset import EmbeddingDataset

def train_classifier_head(
    cfg: Dict[str, Any],
    embeddings,
    labels,
    embedding_dim: int,
    num_classes: int,
    device: str
) -> nn.Module:
    
    csv_file = None
    csv_writer = None
    if cfg["checkpoint"]["train_csv_path"] is not None:
        csv_file, csv_writer = init_csv_logger(cfg["checkpoint"]["train_csv_path"])

    # move data to device
    embeddings = embeddings.to(device) # why not x.to(device) and y.to(device) in mini batch
    labels = labels.to(device)

    # build classifier head
    # classifier = build_vgg_head_deep(in_dim=embedding_dim, num_classes=num_classes).to(device) # VGG
    classifier = ResNet_head(in_dim=embedding_dim, num_classes=num_classes).to(device) # ResNet

    # classifier = LinearClassifier(in_dim=embedding_dim, num_classes=num_classes).to(device) # ResNet
    

    # dataloader
    dataset = EmbeddingDataset(embeddings, labels)
    dataloader = DataLoader(dataset, cfg["training"]["batch_size"], shuffle = True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg["training"]["learning_rate"], momentum=cfg["training"]["momentum"], weight_decay=1e-4)
    
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        classifier.train()

        total_loss = 0.0 # used to store the loss over all samples
        correct = 0
        total = 0

        for x, y in dataloader: # x: [B,C,H,W]; y: [B]
            optimizer.zero_grad()
            logits = classifier(x) # [B, numb_classes]
            loss = criterion(logits, y) # logits will be converted to prob in CrossEntropyLoss
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            total_loss += loss.item() * x.size(0) #x.size(0)=B; loss.item()=the average loss in a batch. total_loss: sum of losses for this batch
            preds = logits.argmax(dim=1) # pick the class that has largest logit preds = [class_i, class_, ...]
            correct += (preds == y).sum().item() # preds == y is [1,0,0,1,...]Bx1 .sum() .item() == number of correct samples
            total += y.size(0) # y.size(0)=B
        avg_loss = total_loss / total # total is the number of samples
        acc = correct / total
        scheduler.step()
        print(
                f"[Head Train] Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}"
            )
        if csv_writer is not None:
            csv_writer.writerow([epoch + 1, avg_loss, acc])
            csv_file.flush() # force the os to write the data in disk
    if csv_file is not None:
        csv_file.close()
    return classifier

def init_csv_logger(csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(csv_path, mode="w", newline="")
    writer = csv.writer(f)
    writer.writerow(["epoch", "avg_loss", "accuracy"])
    f.flush()

    return f, writer
