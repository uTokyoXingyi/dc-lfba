import torch
from torch.utils.data import Subset

# from exp_common import (
#     set_seed, append_csv,
#     default_transform, make_imagefolder, make_loader,
#     build_nested_subsets_imagefolder,
#     EncoderPlusHead,
#     generate_embeddings, train_head_wrapper, evaluate_model,
# )
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.exp_common import set_seed, append_csv, make_imagefolder, make_loader, EncoderPlusHead, build_nested_subsets_imagefolder
from experiments.exp_common import generate_embeddings, train_head_wrapper, evaluate_model, train_model_supervised
# from experiments.exp_common import default_transform

from models.encoder import ResNet18Encoder
from common.utils import load_encoder_checkpoint # function to load trained encoder
from torchvision import datasets, transforms

# -------------------------
# Parameters (edit here)
# -------------------------
SEED = 42
SUBSET_SEED = 0
GPU_ID = 0

DIR_LABELED_TRAIN = "data/client/labeled"
DIR_LABELED_TEST  = "data/evaluation/test"

# N_LIST = [1, 10, 20, 50, 80, 100, 200, 300, 400, 500]
N_LIST = [60]

# Encoder checkpoint (you load your SSL-trained encoder here)
ENCODER_CKPT_SimCLR = "checkpoints/ssl_SimCLR/encoder_epoch_100.pth"
ENCODER_CKPT_BYOL = "checkpoints/ssl_BYOL/encoder_epoch_25.pth"
ENCODER_CKPT_MoCo = "checkpoints/ssl_MoCo/encoder_epoch_3.pth"
ENCODER_CKPT_SimSiam = "checkpoints/ssl_SimSiam/encoder_epoch_3.pth"

# embeddings
# BATCH_EMBED = 64 # labeled data
# NUM_WORKERS = 8
# head

HEAD_EPOCHS = 50
HEAD_LR = 0.01 #0.001
HEAD_WD = 1e-4
momentum = 0.9

# common
# num_classes = 9
IMG_SIZE = 224
batch_size_labeled = 64
batch_size_test = 64
num_workers_labeled = 8
num_workers_test = 8

OUT_CSV_baseline = "runs_E2/exp_label_eff_baseline_more.csv"
OUT_CSV_our_SimCLR = "runs_E2/exp_label_eff_our_SimCLR_more.csv"
OUT_CSV_our_BYOL = "runs_E2/exp_label_eff_our_BYOL.csv"
OUT_CSV_our_MoCo = "runs_E2/exp_label_eff_our_MoCo.csv"
OUT_CSV_our_SimSiam = "runs_E2/exp_label_eff_our_SimSiam.csv"

CSV_FIELDS = [
    "method",
    "N",
    "loss", "s_acc", "b_acc",
]

def default_transform(img_size: int):
    # define the transformation
    resize_size = int(img_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        # transforms.RandomHorizontalFlip(p=0.5), # simulate multiple rooms
        # transforms.RandomApply([
        #             transforms.ColorJitter(
        #                 brightness=0.4,
        #                 contrast=0.4,
        #                 saturation=0.4,
        #                 hue=0.1
        #             )
        #         ], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    set_seed(SEED)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")

    tfm = default_transform(IMG_SIZE)
    ds_train = make_imagefolder(DIR_LABELED_TRAIN, tfm) # for head training
    ds_test  = make_imagefolder(DIR_LABELED_TEST, tfm)
    num_classes = len(ds_train.classes)

    dl_test = make_loader(ds_test, batch_size=batch_size_test, shuffle=False, num_workers=num_workers_test)
    nested = build_nested_subsets_imagefolder(ds_train, N_LIST, seed=SUBSET_SEED)

    # -------------------------
    # SimCLR
    # -------------------------
    encoder_SimCLR = ResNet18Encoder(pretrained=False)
    encoder_SimCLR = load_encoder_checkpoint(ENCODER_CKPT_SimCLR, encoder_SimCLR)
    encoder_SimCLR.to(device)
    encoder_SimCLR.eval()
    for p in encoder_SimCLR.parameters():
        p.requires_grad_(False)
    for N in N_LIST:
        sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
        dl_sub = make_loader(sub_ds, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)
        emb, y = generate_embeddings(encoder_SimCLR, dl_sub, device)

        head = train_head_wrapper(
             embeddings=emb,
             labels=y,
             embedding_dim=emb.shape[1],
             num_classes=num_classes,
             device=device,
             epochs=HEAD_EPOCHS,
             lr=HEAD_LR,
             batch_size = batch_size_labeled,
             momentum = momentum,
        )

        model = EncoderPlusHead(encoder_SimCLR, head).to(device)
        loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
        append_csv(OUT_CSV_our_SimCLR, {
            "method": "ours_fixed_encoder_varN",
            "N": int(N),
            "loss": loss,
            "s_acc": s_acc,
            "b_acc": b_acc,
        }, CSV_FIELDS)
        print(f"labeled images per class={N} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

        # sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
        model_ft = train_model_supervised(
                num_classes=num_classes,
                device=device,
                epochs=50,     # 50/25
                batch_size=batch_size_labeled,
                num_workers=num_workers_labeled,
                lr=0.1,
                weight_decay=1e-4,
                momentum=0.9,
                use_cosine=True,
                pretrained=False, 
                dataset =sub_ds,
            )
        loss, s_acc, b_acc = evaluate_model(model_ft, dl_test, device, num_classes)
        append_csv(OUT_CSV_baseline, {
                "method": "baseline_supervised_eff",
                "N": int(N),
                "loss": loss,
                "s_acc": s_acc,
                "b_acc": b_acc,
            }, CSV_FIELDS)
    
    
    # -------------------------
    # BYOL
    # -------------------------
    # encoder_BYOL = ResNet18Encoder(pretrained=False)
    # encoder_BYOL = load_encoder_checkpoint(ENCODER_CKPT_BYOL, encoder_BYOL)
    # encoder_BYOL.to(device)
    # encoder_BYOL.eval()
    # for p in encoder_BYOL.parameters():
    #     p.requires_grad_(False)
    
    # for N in N_LIST:
    #     sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
    #     dl_sub = make_loader(sub_ds, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)
    #     emb, y = generate_embeddings(encoder_BYOL, dl_sub, device)

    #     head = train_head_wrapper(
    #          embeddings=emb,
    #          labels=y,
    #          embedding_dim=emb.shape[1],
    #          num_classes=num_classes,
    #          device=device,
    #          epochs=HEAD_EPOCHS,
    #          lr=HEAD_LR,
    #          batch_size = batch_size_labeled,
    #          momentum = momentum,
    #     )

    #     model = EncoderPlusHead(encoder_BYOL, head).to(device)
    #     loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
    #     append_csv(OUT_CSV_our_BYOL, {
    #         "method": "ours_fixed_encoder_varN",
    #         "N": int(N),
    #         "loss": loss,
    #         "s_acc": s_acc,
    #         "b_acc": b_acc,
    #     }, CSV_FIELDS)
    #     print(f"labeled images per class={N} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")  

    # -------------------------
    # MoCo
    # -------------------------
    # encoder_MoCo = ResNet18Encoder(pretrained=False)
    # encoder_MoCo = load_encoder_checkpoint(ENCODER_CKPT_MoCo, encoder_MoCo)
    # encoder_MoCo.to(device)
    # encoder_MoCo.eval()
    # for p in encoder_MoCo.parameters():
    #     p.requires_grad_(False)
    # for N in N_LIST:
    #     sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
    #     dl_sub = make_loader(sub_ds, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)
    #     emb, y = generate_embeddings(encoder_MoCo, dl_sub, device)

    #     head = train_head_wrapper(
    #          embeddings=emb,
    #          labels=y,
    #          embedding_dim=emb.shape[1],
    #          num_classes=num_classes,
    #          device=device,
    #          epochs=HEAD_EPOCHS,
    #          lr=HEAD_LR,
    #          batch_size = batch_size_labeled,
    #          momentum = momentum,
    #     )

    #     model = EncoderPlusHead(encoder_MoCo, head).to(device)
    #     loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
    #     append_csv(OUT_CSV_our_MoCo, {
    #         "method": "ours_fixed_encoder_varN",
    #         "N": int(N),
    #         "loss": loss,
    #         "s_acc": s_acc,
    #         "b_acc": b_acc,
    #     }, CSV_FIELDS)
    #     print(f"labeled images per class={N} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")
    
    # -------------------------
    # SimSiam
    # -------------------------
    # encoder_SimSiam = ResNet18Encoder(pretrained=False)
    # encoder_SimSiam = load_encoder_checkpoint(ENCODER_CKPT_SimSiam, encoder_SimSiam)
    # encoder_SimSiam.to(device)
    # encoder_SimSiam.eval()
    # for p in encoder_SimSiam.parameters():
    #     p.requires_grad_(False)
    
    # for N in N_LIST:
    #     sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
    #     dl_sub = make_loader(sub_ds, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)
    #     emb, y = generate_embeddings(encoder_SimSiam, dl_sub, device)

    #     head = train_head_wrapper(
    #          embeddings=emb,
    #          labels=y,
    #          embedding_dim=emb.shape[1],
    #          num_classes=num_classes,
    #          device=device,
    #          epochs=HEAD_EPOCHS,
    #          lr=HEAD_LR,
    #          batch_size = batch_size_labeled,
    #          momentum = momentum,
    #     )

    #     model = EncoderPlusHead(encoder_SimSiam, head).to(device)
    #     loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
    #     append_csv(OUT_CSV_our_SimSiam, {
    #         "method": "ours_fixed_encoder_varN",
    #         "N": int(N),
    #         "loss": loss,
    #         "s_acc": s_acc,
    #         "b_acc": b_acc,
    #     }, CSV_FIELDS)
    #     print(f"labeled images per class={N} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")   
    
    # -------------------------
    # Baseline
    # -------------------------
    # for N in N_LIST:
    #     sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
    #     model_ft = train_model_supervised(
    #             num_classes=num_classes,
    #             device=device,
    #             epochs=50,     # 50/25
    #             batch_size=batch_size_labeled,
    #             num_workers=num_workers_labeled,
    #             lr=0.1,
    #             weight_decay=1e-4,
    #             momentum=0.9,
    #             use_cosine=True,
    #             pretrained=False, 
    #             dataset =sub_ds,
    #         )
    #     loss, s_acc, b_acc = evaluate_model(model_ft, dl_test, device, num_classes)
    #     append_csv(OUT_CSV_baseline, {
    #             "method": "baseline_supervised_eff",
    #             "N": int(N),
    #             "loss": loss,
    #             "s_acc": s_acc,
    #             "b_acc": b_acc,
    #         }, CSV_FIELDS)
    print("Done.")

if __name__ == "__main__":
    main()