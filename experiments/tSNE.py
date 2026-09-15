import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch.utils.data import Subset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.exp_common import set_seed, default_transform, make_imagefolder, make_loader
from experiments.exp_common import generate_embeddings

from models.encoder import ResNet18Encoder
from common.utils import load_encoder_checkpoint # function to load trained encoder


# -------------------------
# Parameters (edit here)
# -------------------------
SEED = 42
SUBSET_SEED = 0
GPU_ID = 1

DIR_LABELED_TRAIN = "data/client/labeled"
DIR_LABELED_TEST  = "data/evaluation/test"


# Encoder checkpoint (you load your SSL-trained encoder here)
ENCODER_CKPT_SimCLR = "checkpoints/ssl_SimCLR/encoder_epoch_200.pth"
# ENCODER_CKPT_SimCLR = "checkpoints/ssl_MoCo/encoder_epoch_3.pth"

# common
IMG_SIZE = 224
batch_size_labeled = 64
batch_size_test = 64
num_workers_labeled = 8
num_workers_test = 8

OUT_DIR = "runs_E3"
# OUT_CSV_our_SimCLR = "runs_E3/exp_tSNR_our_SimCLR.csv"
OUT_FIG = os.path.join(OUT_DIR, "tsne_random_vs_simclr_200.png")
# t-SNE params
TSNE_PERPLEXITY = 30
TSNE_LR = 200
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42

def run_tsne(features):
    """
    features: numpy array of shape [N, D]
    return: numpy array of shape [N, 2]
    """
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LR,
        max_iter=TSNE_N_ITER,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
    )
    return tsne.fit_transform(features)

def plot_tsne(ax, features_2d, labels, class_names, title):
    """
    ax: matplotlib axis
    features_2d: [N, 2]
    labels: [N]
    class_names: list of class names
    """
    num_classes = len(class_names)

    for c in range(num_classes):
        idx = labels == c
        ax.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=8,
            alpha=0.7,
            label=class_names[c]
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

def main():
    set_seed(SEED)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")

    os.makedirs(OUT_DIR, exist_ok=True)

    tfm = default_transform(IMG_SIZE)
    ds_train = make_imagefolder(DIR_LABELED_TRAIN, tfm) # load labeled dataset, for our head training
    class_names = ds_train.classes
    num_classes = len(ds_train.classes)

    dl_train = make_loader(ds_train, batch_size=batch_size_labeled, shuffle=False, num_workers=num_workers_labeled)

    # -------------------------
    # 1) Random initialization encoder
    # -------------------------
    encoder_random = ResNet18Encoder(pretrained=False)
    encoder_random.to(device)
    encoder_random.eval()
    for p in encoder_random.parameters():
        p.requires_grad_(False)

    emb_random, y_random = generate_embeddings(encoder_random, dl_train, device)


    # -------------------------
    # SimCLR Encoder
    # -------------------------
    encoder_simclr = ResNet18Encoder(pretrained=True)
    # encoder_simclr = load_encoder_checkpoint(ENCODER_CKPT_SimCLR, encoder_simclr)
    encoder_simclr.to(device)
    encoder_simclr.eval()
    for p in encoder_simclr.parameters():
        p.requires_grad_(False)
    
    emb_simclr, y_simclr = generate_embeddings(encoder_simclr, dl_train, device)

    if isinstance(emb_random, torch.Tensor):
        emb_random = emb_random.cpu().numpy()
    if isinstance(y_random, torch.Tensor):
        y_random = y_random.cpu().numpy()

    if isinstance(emb_simclr, torch.Tensor):
        emb_simclr = emb_simclr.cpu().numpy()
    if isinstance(y_simclr, torch.Tensor):
        y_simclr = y_simclr.cpu().numpy()

    # sanity check
    assert np.array_equal(y_random, y_simclr), "Label order mismatch between random and SimCLR embeddings."

    # -------------------------
    # 3) Run t-SNE separately
    # -------------------------
    tsne_random = run_tsne(emb_random)
    tsne_simclr = run_tsne(emb_simclr)

    # -------------------------
    # 4) Plot side-by-side
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_tsne(
        axes[0],
        tsne_random,
        y_random,
        class_names,
        title="Random Initialization Encoder"
    )

    plot_tsne(
        axes[1],
        tsne_simclr,
        y_simclr,
        class_names,
        title="SimCLR Encoder (100 SSL epochs)"
    )

    # show legend only once
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5))
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    # plt.savefig(OUT_FIG, bbox_inches="tight")

    plt.show()

    print(f"Saved figure to: {OUT_FIG}")
    
    print("Done.")

if __name__ == "__main__":
    main()