import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time

from experiments.exp_common import set_seed, append_csv, default_transform, make_imagefolder, make_loader, EncoderPlusHead
from experiments.exp_common import generate_embeddings, train_head_wrapper, evaluate_model, train_encoder_contrastive_wrapper, train_model_supervised


from common.utils import save_encoder_checkpoint
from models.encoder import ResNet18Encoder, ResNet50Encoder, build_vgg19bn_encoder
from common.utils import load_encoder_checkpoint # function to load trained encoder

from pathlib import Path
# -------------------------
# Parameters (edit here)
# -------------------------
SEED = 42
GPU_ID = 1

DIR_UNLABELED = "data/client/unlabeled"

DIR_LABELED_TRAIN = "data/client/labeled"

DIR_LABELED_TEST  = "data/evaluation/test"

# ENCODER_CKPT = "checkpoints/ssl_SimCLR/encoder_epoch_100_GPU1.pth"

OUT_CSV_M1 = "runs_E4/exp_11_time_SimCLR.csv"
OUT_CSV_B = "runs_E4/exp_11_time_Baseline.csv"
OUT_CSV_TIME = "runs_E4/exp_11_training_time_Baseline.csv"
CSV_FIELDS = [
    "method",
    "ssl_epochs",
    "loss", "s_acc", "b_acc",
]
CSV_FIELDS_Baseline = [
    "method",
    "sup_epochs",
    "loss", "s_acc", "b_acc",
]
CSV_FIELDS_TIME = [
    "method",
    "epochs",
    "total_time_sec",
]
# our method
## ssl
# EPOCHS_SSL_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100]
EPOCHS_SUP_LIST = [100]
batch_size_ssl = 512
num_workers_ssl = 8
learning_rate_ssl = 0.3
weight_decay_ssl = 1e-4
temperature = 0.1

momentum = 0.9

save_dir_M1 = "checkpoints/ssl_SimCLR"


ssl_csv_path_M1 = "checkpoints/ssl_SimCLR/ssl_metrics.csv"

save_interval = 50
# projection_dim = 128
log_interval = 20

## embedding generating and also for head training and baseline
# batch_size_emb = 64
# num_workers_emb = 8

## head training
HEAD_EPOCHS = 50
HEAD_LR = 0.01 #0.001
HEAD_WD = 1e-5

# baseline - supervised
# EPOCHS_SUP_SCRATCH = 100

## evaluation
# batch_size_eval = 64
# number_workers_eval = 8

# common
# num_classes = 9
IMG_SIZE = 224
batch_size_labeled = 64
batch_size_test = 64
num_workers_labeled = 8
num_workers_test = 8

# ------------------------
# helper
# ------------------------
def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def main():

    set_seed(SEED)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
    # print(device)

    # dataset
    tfm = default_transform(IMG_SIZE)
    ds_train = make_imagefolder(DIR_LABELED_TRAIN, tfm) # load labeled dataset, for our head training
    ds_test  = make_imagefolder(DIR_LABELED_TEST, tfm) # load test dataset
    num_classes = len(ds_train.classes)

    dl_train = make_loader(ds_train, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled) # loader will be used in our method and baseline
    dl_test = make_loader(ds_test, batch_size=batch_size_test, shuffle=False, num_workers=num_workers_test)

    # -------------------------
    # Our SSL encoder training
    # -------------------------
    # for e_ssl in EPOCHS_SSL_LIST:
    #     print(f"Running Epoch {e_ssl}")
    #     # -------------------------
    #     # 1. Stage-I SSL training
    #     # -------------------------
    #     sync_device(device)
    #     start = time.perf_counter()

    #     if e_ssl == 0:
    #         encoder = ResNet18Encoder(pretrained=True)
    #     else:
    #         encoder = train_encoder_contrastive_wrapper(
    #             method = "simclr", # {simclr, moco, byol, simsiam}
    #             epochs_ssl = e_ssl,
    #             batch_size = 512,
    #             num_workers= num_workers_ssl,
    #             lr = 0.6,
    #             weight_decay = weight_decay_ssl,
    #             temperature = temperature,
    #             momentum = momentum,
    #             dir_unlabeled = DIR_UNLABELED,
    #             image_size = IMG_SIZE,
    #             save_dir = save_dir_M1,
    #             save_interval = save_interval,
    #             ssl_csv_path = ssl_csv_path_M1,
    #             projection_dim = 128,
    #             log_interval = log_interval,
    #             device = device,
    #         )
    #     sync_device(device)
    #     t_ssl = time.perf_counter() - start

    #     save_encoder_checkpoint(save_dir_M1,encoder,e_ssl)
    #     encoder.to(device)
    #     encoder.eval()
    #     for p in encoder.parameters():
    #         p.requires_grad = False
        
    #     # =========================
    #     # 2. Embedding generation
    #     # =========================
    #     sync_device(device)
    #     start = time.perf_counter()
    #     emb_train, y_train = generate_embeddings(encoder, dl_train, device)
    #     sync_device(device)
    #     t_emb = time.perf_counter() - start

    #     print(f"[Server] Embedding dim: {emb_train.shape[1]}")
    #     # =========================
    #     # 3. Stage-II head training
    #     # =========================
    #     sync_device(device)
    #     start = time.perf_counter()
    #     head = train_head_wrapper(
    #          embeddings=emb_train,
    #          labels=y_train,
    #          embedding_dim=emb_train.shape[1],
    #          num_classes=num_classes,
    #          device=device,
    #          batch_size = batch_size_labeled,
    #          momentum = momentum,
    #          epochs=HEAD_EPOCHS,
    #          lr=HEAD_LR,
    #     )
    #     sync_device(device)
    #     t_head = time.perf_counter() - start
    #     # =========================
    #     # Total computational time
    #     # =========================
    #     t_total = t_ssl + t_emb + t_head
    #     print("\n========== DC-LFBA Timing ==========")
    #     print(f"SSL training:         {t_ssl:.2f} s ({t_ssl/60:.2f} min)")
    #     print(f"Embedding generation: {t_emb:.2f} s ({t_emb/60:.2f} min)")
    #     print(f"Head training:        {t_head:.2f} s ({t_head/60:.2f} min)")
    #     print(f"Total:                {t_total:.2f} s ({t_total/60:.2f} min)")
    #     print("====================================\n")
    #     append_csv(
    #         OUT_CSV_TIME,
    #         {
    #             "method": "DC-LFBA",
    #             "ssl_epochs": e_ssl,
    #             "head_epochs": HEAD_EPOCHS,
    #             "ssl_time_sec": t_ssl,
    #             "embedding_time_sec": t_emb,
    #             "head_time_sec": t_head,
    #             "total_time_sec": t_total,
    #         },
    #         CSV_FIELDS_TIME,
    #     )

    #     if e_ssl == 0:
    #         encoder = ResNet18Encoder(pretrained=True)
        
    #     model = EncoderPlusHead(encoder, head)
    #     model.to(device)
    #     loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
        
    #     append_csv(OUT_CSV_M1, {
    #         "method": "ours_ssl_then_head",
    #         "ssl_epochs": e_ssl,
    #         "sup_epochs": "",
    #         "loss": loss,
    #         "s_acc": s_acc,
    #         "b_acc": b_acc,
    #     }, CSV_FIELDS)
        
    #     print(f"SSL epoch={e_ssl} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

    # -------------------------
    # Supervised baseline from scratch (end-to-end)
    # -------------------------
    for e_sup in EPOCHS_SUP_LIST:
        sync_device(device)
        start = time.perf_counter()
        model_sup = train_model_supervised(
             num_classes=num_classes,
             device=device,
             epochs=e_sup,
             batch_size=batch_size_labeled,
             num_workers=num_workers_labeled,
             lr=0.1,
             weight_decay=1e-4,
             momentum=0.9,
             use_cosine=True,
             pretrained=False,
             dir_labeled_train=DIR_LABELED_TRAIN,
             transform=tfm,
            )
        sync_device(device)
        t_super = time.perf_counter() - start
        print("\n===== Supervised LFBA Timing =====")
        print(f"Total: {t_super:.2f} s ({t_super/60:.2f} min)")
        print("====================================\n")
        append_csv(
            OUT_CSV_TIME,
            {
                "method": "supervised_LFBA",
                "epochs": e_sup,
                "total_time_sec": t_super,
            },
            CSV_FIELDS_TIME,
        )
        model_sup.to(device)
        loss, s_acc, b_acc = evaluate_model(model_sup, dl_test, device, num_classes)
        append_csv(OUT_CSV_B, {
            "method": "baseline_supervised_e2e_scratch",
            "sup_epochs": e_sup,
            "loss": loss,
            "s_acc": s_acc,
            "b_acc": b_acc,
            }, CSV_FIELDS_Baseline
         )
        print(f"supervised epoch={e_sup} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

    print(f"Done")

if __name__ == "__main__":
    main()
