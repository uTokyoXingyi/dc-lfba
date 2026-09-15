# exp_12_pretrained.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.exp_common import set_seed, append_csv,  default_transform, make_imagefolder, make_loader,  EncoderPlusHead
from experiments.exp_common import generate_embeddings, train_head_wrapper, evaluate_model,  train_model_supervised

# from common.utils import save_encoder_checkpoint
from models.encoder import ResNet18Encoder, ResNet50Encoder, build_vgg19bn_encoder

# from evaluation.eval import evaluate

SEED = 42
GPU_ID = 1

# DIR_UNLABELED = "data/client/unlabeled"
DIR_LABELED_TRAIN = "data/client/labeled"
DIR_LABELED_TEST  = "data/evaluation/test"

# Baseline: pretrained full fine-tune
# EPOCHS_FINETUNE = 50

# our method - head training
# HEAD_EPOCHS = 50
HEAD_LR = 0.01
HEAD_WD = 1e-4
momentum = 0.9

# common
IMG_SIZE = 224
batch_size_labeled = 64
batch_size_test = 64
num_workers_labeled = 8
num_workers_test = 8
EPOCHS_COMMON = [50]
# EPOCHS_COMMON = [1, 2]


OUT_CSV_our = "runs_E1/exp_12_pretrained_our.csv"
OUT_CSV_baseline = "runs_E1/exp_12_pretrained_baseline.csv"

CSV_FIELDS = [
    "method",
    "epochs",
    "loss", "s_acc", "b_acc",
]

def main():
    set_seed(SEED)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
    print(device)
    tfm = default_transform(IMG_SIZE)
    ds_train = make_imagefolder(DIR_LABELED_TRAIN, tfm)
    ds_test  = make_imagefolder(DIR_LABELED_TEST, tfm)
    num_classes = len(ds_train.classes)

    dl_train = make_loader(ds_train, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)
    dl_test = make_loader(ds_test, batch_size=batch_size_test, shuffle=False, num_workers=num_workers_test)
    class_index = ds_test.class_to_idx
    print("Class mapping:", class_index)
    # -------------------------
    # Ours deployment: pretrained frozen encoder + head
    # -------------------------
    encoder_pt = ResNet18Encoder(pretrained=True)
    encoder_pt.to(device)
    encoder_pt.eval()
    for p in encoder_pt.parameters():
        p.requires_grad_(False)

    emb_train, y_train = generate_embeddings(encoder_pt, dl_train, device)

    for epoch in EPOCHS_COMMON:
        # head = train_head_wrapper(
        #         embeddings=emb_train,
        #         labels=y_train,
        #         embedding_dim=emb_train.shape[1],
        #         num_classes=num_classes,
        #         device=device,
        #         epochs=epoch,
        #         lr=HEAD_LR,
        #         batch_size = batch_size_labeled,
        #         momentum = momentum,
        #     )
        # # encoder_pt_1 = ResNet18Encoder(pretrained=False)
        # model = EncoderPlusHead(encoder_pt, head)
        # model.to(device)
        # model.eval()
        # loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)

        # append_csv(OUT_CSV_our, {
        #     "method": "pretrained_frozen_plus_head",
        #     "epochs": epoch,
        #     "loss": loss,
        #     "s_acc": s_acc,
        #     "b_acc": b_acc,
        # }, CSV_FIELDS)
        # print("pre-train_our_method")
        # print(f"Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")
        # -------------------------
        # Baseline: pretrained full fine-tuning end-to-end
        # -------------------------
        model_ft = train_model_supervised(
            num_classes=num_classes,
            device=device,
            epochs=epoch,
            batch_size=batch_size_labeled,
            num_workers=num_workers_labeled,
            lr=0.1,
            weight_decay=1e-4,
            momentum=0.9,
            use_cosine=False,
            pretrained=True,
            dir_labeled_train=DIR_LABELED_TRAIN,
            transform=tfm,
        )
        model_ft.to(device)

        loss, s_acc, b_acc = evaluate_model(model_ft, dl_test, device, num_classes)

        append_csv(OUT_CSV_baseline, {
            "method": "pretrained_full_finetune",
            "epochs": epoch,
            "loss": loss,
            "s_acc": s_acc,
            "b_acc": b_acc,
        }, CSV_FIELDS)
        print("pre-train_baseline")
        print(f"Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

    print("Done")

if __name__ == "__main__":
    main()