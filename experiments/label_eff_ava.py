import torch
from torch.utils.data import Subset
import numpy as np

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
SEED = 42 # this is used for model
SUBSET_SEED_LIST = [0, 1, 2, 3, 4] # this is used for random select N images
GPU_ID = 0

DIR_LABELED_TRAIN = "data/client/labeled"
DIR_LABELED_TEST  = "data/evaluation/test"

N_LIST = [1, 10, 20, 50, 80, 100, 200, 300, 400, 500]
# N_LIST = [150, 500]

# Encoder checkpoint (you load your SSL-trained encoder here)
ENCODER_CKPT = "checkpoints/ssl_SimCLR/encoder_epoch_100_GPU0.pth"


HEAD_EPOCHS = 50
HEAD_LR = 0.01 #0.001
HEAD_WD = 1e-4
momentum = 0.9

# common
IMG_SIZE = 224
batch_size_labeled = 64
batch_size_test = 64
num_workers_labeled = 2
num_workers_test = 2

OUT_CSV_our = "runs_E2/label_eff_our_ava.csv"
OUT_CSV_baseline = "runs_E2/label_eff_baseline_ava.csv"
CSV_FIELDS = [
    "method",
    "N",
    "loss", "s_acc", "b_acc",
]

def default_transform(img_size: int):
    # define the transformation
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

def main():
    # set_seed(SEED)
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

    encoder = ResNet18Encoder(pretrained=False)
    encoder = load_encoder_checkpoint(ENCODER_CKPT, encoder)

    # encoder = ResNet18Encoder(pretrained=True)

    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for N in N_LIST:
        loss_list_our = []
        s_acc_list_our = []
        b_acc_list_our = []

        loss_list_sup = []
        s_acc_list_sup = []
        b_acc_list_sup = []

        for SUBSET_SEED in SUBSET_SEED_LIST:
            print(f"Running seed {SUBSET_SEED}")
            set_seed(SEED)
            nested = build_nested_subsets_imagefolder(ds_train, N_LIST, seed=SUBSET_SEED)
            sub_ds = Subset(ds_train, nested[N]) # sub_dataset: different N
            dl_sub = make_loader(sub_ds, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)
            emb, y = generate_embeddings(encoder, dl_sub, device)

            head = train_head_wrapper(
                embeddings=emb,
                labels=y,
                embedding_dim=emb.shape[1],
                num_classes=num_classes,
                device=device,
                epochs=HEAD_EPOCHS,
                lr=0.01,
                batch_size = batch_size_labeled,
                momentum = momentum,
            )

            model = EncoderPlusHead(encoder, head).to(device)
            loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
            loss_list_our.append(loss)
            s_acc_list_our.append(s_acc)
            b_acc_list_our.append(b_acc)

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
            loss_list_sup.append(loss)
            s_acc_list_sup.append(s_acc)
            b_acc_list_sup.append(b_acc)

        print(len(SUBSET_SEED_LIST))
        loss_mean = np.mean(loss_list_our) 
        s_acc_mean = np.mean(s_acc_list_our)
        b_acc_mean = np.mean(b_acc_list_our)
        append_csv(OUT_CSV_our, {
            "method": "ours_SimCLR",
            "N": int(N),
            "loss": loss_mean,
            "s_acc": s_acc_mean,
            "b_acc": b_acc_mean,
        }, CSV_FIELDS)
        print(f"labeled images per class={N} | Loss: {loss_mean:.4f} | S-Acc: {s_acc_mean:.4f} | B-Acc: {b_acc_mean:.4f}")   

        loss_mean = np.mean(loss_list_sup) 
        s_acc_mean = np.mean(s_acc_list_sup)
        b_acc_mean = np.mean(b_acc_list_sup)
        append_csv(OUT_CSV_baseline, {
                "method": "baseline_supervised_eff",
                "N": int(N),
                "loss": loss_mean,
                "s_acc": s_acc_mean,
                "b_acc": b_acc_mean,
            }, CSV_FIELDS)
        print(f"labeled images per class={N} | Loss: {loss_mean:.4f} | S-Acc: {s_acc_mean:.4f} | B-Acc: {b_acc_mean:.4f}")   
    print("Done.")

if __name__ == "__main__":
    main()