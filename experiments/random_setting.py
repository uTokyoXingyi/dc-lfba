import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
GPU_ID = 0

DIR_UNLABELED = "data/client/unlabeled"

DIR_LABELED_TRAIN = "data/client/labeled"

DIR_LABELED_TEST  = "data/evaluation/test"

ENCODER_CKPT = "checkpoints/ssl_SimCLR/encoder_epoch_100_GPU1.pth"


OUT_CSV_M1 = "runs_E1/exp_11_random_init_SimCLR.csv"
OUT_CSV_M2 = "runs_E1/exp_11_random_init_MoCo.csv"
OUT_CSV_M3 = "runs_E1/exp_11_random_init_BYOL.csv"
OUT_CSV_M4 = "runs_E1/exp_11_random_init_SimSiam.csv"

OUT_CSV_B = "runs_E1/exp_11_random_init_Baseline_0.1.csv"
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
# our method
## ssl
EPOCHS_SSL_LIST = [1, 2, 3, 4, 5, 10, 25, 50, 100]
batch_size_ssl = 512
num_workers_ssl = 8
learning_rate_ssl = 0.3
weight_decay_ssl = 1e-4
temperature = 0.1

momentum = 0.9

save_dir_M1 = "checkpoints/ssl_SimCLR"
save_dir_M2 = "checkpoints/ssl_MoCo"
save_dir_M3 = "checkpoints/ssl_BYOL"
save_dir_M4 = "checkpoints/ssl_SimSiam"


ssl_csv_path_M1 = "checkpoints/ssl_SimCLR/ssl_metrics.csv"
ssl_csv_path_M2 = "checkpoints/ssl_MoCo/ssl_metrics.csv"
ssl_csv_path_M3 = "checkpoints/ssl_BYOL/ssl_metrics.csv"
ssl_csv_path_M4 = "checkpoints/ssl_SimSiam/ssl_metrics.csv"

save_interval = 50
# projection_dim = 128
log_interval = 20

## embedding generating and also for head training and baseline
# batch_size_emb = 64
# num_workers_emb = 8

## head training
HEAD_EPOCHS = 25
HEAD_LR = 0.01
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
    for e_ssl in EPOCHS_SSL_LIST:
        print(f"Running Epoch {e_ssl}")
        # -------------------------
        # SimCLR
        # -------------------------
        if e_ssl == 0:
            encoder = ResNet18Encoder(pretrained=True)
        else:
            encoder = train_encoder_contrastive_wrapper(
                method = "simclr", # {simclr, moco, byol, simsiam}
                epochs_ssl = e_ssl,
                batch_size = 512,
                num_workers= num_workers_ssl,
                lr = 0.3,
                weight_decay = weight_decay_ssl,
                temperature = temperature,
                momentum = momentum,
                dir_unlabeled = DIR_UNLABELED,
                image_size = IMG_SIZE,
                save_dir = save_dir_M1,
                save_interval = save_interval,
                ssl_csv_path = ssl_csv_path_M1,
                projection_dim = 128,
                log_interval = log_interval,
                device = device,
            )
        save_encoder_checkpoint(save_dir_M1,encoder,e_ssl)
        # test
        # encoder = ResNet18Encoder(pretrained=False)
        # encoder = load_encoder_checkpoint(ENCODER_CKPT, encoder)

        encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        
        emb_train, y_train = generate_embeddings(encoder, dl_train, device)
        print(f"[Server] Embedding dim: {emb_train.shape[1]}")
        head = train_head_wrapper(
             embeddings=emb_train,
             labels=y_train,
             embedding_dim=emb_train.shape[1],
             num_classes=num_classes,
             device=device,
             batch_size = batch_size_labeled,
             momentum = momentum,
             epochs=HEAD_EPOCHS,
             lr=HEAD_LR,
        )
        if e_ssl == 0:
            encoder = ResNet18Encoder(pretrained=True)
        
        model = EncoderPlusHead(encoder, head)
        model.to(device)
        loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
        
        append_csv(OUT_CSV_M1, {
            "method": "ours_ssl_then_head",
            "ssl_epochs": e_ssl,
            "sup_epochs": "",
            "loss": loss,
            "s_acc": s_acc,
            "b_acc": b_acc,
        }, CSV_FIELDS)
        
        print(f"SSL epoch={e_ssl} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")
        
        # test load
        # encoder1 = ResNet18Encoder(pretrained=False)
        # encoder1 = load_encoder_checkpoint(ENCODER_CKPT, encoder1)

        # encoder1.to(device)
        # encoder1.eval()
        # for p in encoder1.parameters():
        #     p.requires_grad = False
        
        # emb_train, y_train = generate_embeddings(encoder1, dl_train, device)
        # print(f"[Server] Embedding dim: {emb_train.shape[1]}")
        # head1 = train_head_wrapper(
        #      embeddings=emb_train,
        #      labels=y_train,
        #      embedding_dim=emb_train.shape[1],
        #      num_classes=num_classes,
        #      device=device,
        #      batch_size = batch_size_labeled,
        #      momentum = momentum,
        #      epochs=HEAD_EPOCHS,
        #      lr=HEAD_LR,
        # )
        # if e_ssl == 0:
        #     encoder1 = ResNet18Encoder(pretrained=True)
        
        # model1 = EncoderPlusHead(encoder1, head1)
        # model1.to(device)
        # loss, s_acc, b_acc = evaluate_model(model1, dl_test, device, num_classes)
        
        # append_csv(OUT_CSV_M1, {
        #     "method": "ours_ssl_then_head",
        #     "ssl_epochs": e_ssl,
        #     "sup_epochs": "",
        #     "loss": loss,
        #     "s_acc": s_acc,
        #     "b_acc": b_acc,
        # }, CSV_FIELDS)

        # print(f"Finsh SimCLR for {e_ssl}")

        # -------------------------
        # MoCo
        # -------------------------
        # if e_ssl == 0:
        #     encoder = ResNet18Encoder(pretrained=False)
        # else:
        #     encoder = train_encoder_contrastive_wrapper(
        #             method = "moco", # {simclr, moco, byol, simsiam}
        #             epochs_ssl = e_ssl,
        #             batch_size = 512,
        #             num_workers= num_workers_ssl,
        #             lr = 0.3,
        #             weight_decay = weight_decay_ssl,
        #             temperature = 0.2,
        #             momentum = momentum,
        #             dir_unlabeled = DIR_UNLABELED,
        #             image_size = IMG_SIZE,
        #             save_dir = save_dir_M2, # we doesn't use it actually
        #             save_interval = save_interval,
        #             ssl_csv_path = ssl_csv_path_M2,
        #             projection_dim = 256,
        #             log_interval = log_interval,
        #             device = device,
        #         )
        # encoder.to(device)
        # encoder.eval()
        # for p in encoder.parameters():
        #     p.requires_grad = False
        # # encoder.train()
        # emb_train, y_train = generate_embeddings(encoder, dl_train, device)
        # print(f"[Server] Embedding dim: {emb_train.shape[1]}")
        # head = train_head_wrapper(
        #      embeddings=emb_train,
        #      labels=y_train,
        #      embedding_dim=emb_train.shape[1],
        #      num_classes=num_classes,
        #      device=device,
        #      batch_size = batch_size_labeled,
        #      momentum = momentum,
        #      epochs=HEAD_EPOCHS,
        #      lr=HEAD_LR,
        # )
        # if e_ssl == 0:
        #     encoder = ResNet18Encoder(pretrained=False)
        
        # model = EncoderPlusHead(encoder, head)
        # model.to(device)
        # loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
        # save_encoder_checkpoint(save_dir_M2,encoder,e_ssl)
        # append_csv(OUT_CSV_M2, {
        #     "method": "ours_ssl_then_head",
        #     "ssl_epochs": e_ssl,
        #     "sup_epochs": "",
        #     "loss": loss,
        #     "s_acc": s_acc,
        #     "b_acc": b_acc,
        # }, CSV_FIELDS)
        
        # print(f"SSL epoch={e_ssl} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")
        # print(f"Finsh MoCo for {e_ssl}")

        # -------------------------
        # byol
        # -------------------------
        # if e_ssl == 0:
        #     encoder = ResNet18Encoder(pretrained=False)
        # else:
        #     encoder = train_encoder_contrastive_wrapper(
        #             method = "byol", # {simclr, moco, byol, simsiam}
        #             epochs_ssl = e_ssl,
        #             batch_size = 256,
        #             num_workers= num_workers_ssl,
        #             lr = 0.002,
        #             weight_decay = weight_decay_ssl,
        #             temperature = temperature,
        #             momentum = momentum,
        #             dir_unlabeled = DIR_UNLABELED,
        #             image_size = IMG_SIZE,
        #             save_dir = save_dir_M3,
        #             save_interval = save_interval,
        #             ssl_csv_path = ssl_csv_path_M3,
        #             projection_dim = 256,
        #             log_interval = log_interval,
        #             device = device,
        #         )
        # encoder.to(device)
        # encoder.eval()
        # for p in encoder.parameters():
        #     p.requires_grad = False

        # emb_train, y_train = generate_embeddings(encoder, dl_train, device)
        # print(f"[Server] Embedding dim: {emb_train.shape[1]}")
        # head = train_head_wrapper(
        #      embeddings=emb_train,
        #      labels=y_train,
        #      embedding_dim=emb_train.shape[1],
        #      num_classes=num_classes,
        #      device=device,
        #      batch_size = batch_size_labeled,
        #      momentum = momentum,
        #      epochs=HEAD_EPOCHS,
        #      lr=HEAD_LR,
        # )
        # if e_ssl == 0:
        #     encoder = ResNet18Encoder(pretrained=False)
        
        # model = EncoderPlusHead(encoder, head)
        # model.to(device)
        # loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
        # save_encoder_checkpoint(save_dir_M3,encoder,e_ssl)
        # append_csv(OUT_CSV_M3, {
        #     "method": "ours_ssl_then_head",
        #     "ssl_epochs": e_ssl,
        #     "sup_epochs": "",
        #     "loss": loss,
        #     "s_acc": s_acc,
        #     "b_acc": b_acc,
        # }, CSV_FIELDS)
        
        # print(f"SSL epoch={e_ssl} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")
        # print(f"Finsh BYOL for {e_ssl}")

        # # -------------------------
        # # SimSiam
        # # -------------------------
        # encoder = train_encoder_contrastive_wrapper(
        #         method = "simsiam", # {simclr, moco, byol, simsiam}
        #         epochs_ssl = e_ssl,
        #         batch_size = 256,
        #         num_workers= num_workers_ssl,
        #         lr = 0.1,
        #         weight_decay = weight_decay_ssl,
        #         temperature = temperature,
        #         momentum = momentum,
        #         dir_unlabeled = DIR_UNLABELED,
        #         image_size = IMG_SIZE,
        #         save_dir = save_dir_M4,
        #         save_interval = save_interval,
        #         ssl_csv_path = ssl_csv_path_M4,
        #         projection_dim = 256,
        #         log_interval = log_interval,
        #         device = device,
        #     )
        # encoder.to(device)
        # encoder.eval()
        # for p in encoder.parameters():
        #     p.requires_grad = False

        # emb_train, y_train = generate_embeddings(encoder, dl_train, device)
        # print(f"[Server] Embedding dim: {emb_train.shape[1]}")
        # head = train_head_wrapper(
        #      embeddings=emb_train,
        #      labels=y_train,
        #      embedding_dim=emb_train.shape[1],
        #      num_classes=num_classes,
        #      device=device,
        #      batch_size = batch_size_labeled,
        #      momentum = momentum,
        #      epochs=HEAD_EPOCHS,
        #      lr=HEAD_LR,
        # )
        # if e_ssl == 0:
        #     encoder = ResNet18Encoder(pretrained=False)
        
        # model = EncoderPlusHead(encoder, head)
        # model.to(device)
        # loss, s_acc, b_acc = evaluate_model(model, dl_test, device, num_classes)
        # save_encoder_checkpoint(save_dir_M4,encoder,e_ssl)
        # append_csv(OUT_CSV_M4, {
        #     "method": "ours_ssl_then_head",
        #     "ssl_epochs": e_ssl,
        #     "sup_epochs": "",
        #     "loss": loss,
        #     "s_acc": s_acc,
        #     "b_acc": b_acc,
        # }, CSV_FIELDS)
        
        # print(f"SSL epoch={e_ssl} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")
        # print(f"Finsh SimSiam for {e_ssl}")

    # -------------------------
    # Supervised baseline from scratch (end-to-end)
    # -------------------------
    # for e_sup in EPOCHS_SSL_LIST:
    #     model_sup = train_model_supervised(
    #          num_classes=num_classes,
    #          device=device,
    #          epochs=e_sup,
    #          batch_size=batch_size_labeled,
    #          num_workers=num_workers_labeled,
    #          lr=0.1,
    #          weight_decay=1e-4,
    #          momentum=0.9,
    #          use_cosine=True,
    #          pretrained=False,
    #          dir_labeled_train=DIR_LABELED_TRAIN,
    #          transform=tfm,
    #         )
    #     model_sup.to(device)
    #     loss, s_acc, b_acc = evaluate_model(model_sup, dl_test, device, num_classes)
    #     append_csv(OUT_CSV_B, {
    #         "method": "baseline_supervised_e2e_scratch",
    #         "sup_epochs": e_sup,
    #         "loss": loss,
    #         "s_acc": s_acc,
    #         "b_acc": b_acc,
    #         }, CSV_FIELDS_Baseline
    #      )
    #     print(f"supervised epoch={e_sup} | Loss: {loss:.4f} | S-Acc: {s_acc:.4f} | B-Acc: {b_acc:.4f}")

    print(f"Done")

if __name__ == "__main__":
    main()
