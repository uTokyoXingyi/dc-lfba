# DC-LFBA: Distributed Contrastive Learning for Logic-Free Building Automation
**Contrastive Learning Based Logic-Free Building Automation: A Distributed Approach**

author 1, author 2 — The University of Tokyo

*Accepted at IEEE WFIoT 2026. Paper to appear — citation details will be added once published.*

## Overview


Logic-Free Building Automation (LFBA) learns building control policies directly from camera images and switch-state history, instead of manually designed rules. Prior approaches (e.g. split learning-based LFBA) require exchanging intermediate activations and gradients at every training step, which is costly in bandwidth-constrained IoT settings, and rely on large amounts of labeled data.

**DC-LFBA** decouples representation learning from task-specific prediction:

1. **Stage I — Local encoder training:** a camera node trains a visual encoder using self-supervised contrastive learning(SimCLR, with MoCo/BYOL/SimSiam also supported) on unlabeled images, entirely on-device.
2. **Stage II — Task-specific head training:** the frozen encoder generates compact embeddings from a small labeled dataset, which are sent to a controller node **once** to train a lightweight prediction head.

This requires only a single round of communication (rather than one pertraining iteration) and performs competitively with fully supervised LFBA, while notably outperforming it in low-label regimes.

> **Note:** this repository implements and evaluates the training pipeline
> (Stages I & II) with the client/controller split simulated locally on a
> single machine. Real network communication between devices is not
> implemented in this version.

## Project Structure
```
.
├── client/ # Camera-node logic (Stage I & II, client side)
│ ├── client.py
│ ├── communicator.py
│ ├── datasets.py
│ ├── embedding_generator.py
│ └── ssl_trainer.py
├── server/ # Controller-node logic (Stage II, server side)
│ ├── server.py
│ ├── receiver.py
│ ├── EmbDataset.py
│ └── head_trainer.py
├── models/ # Model architectures shared by client and server
│ ├── encoder.py
│ ├── projection_head.py
│ ├── classifier_head.py
│ ├── BYOLPredictor.py
│ └── SimSiamPredictor.py
├── common/ # Shared utilities used by both client and server
│ └── utils.py
├── evaluation/ # Evaluation pipeline (accuracy, balanced accuracy)
│ ├── dataloader.py
│ └── eval.py
├── configs/ # YAML configs for each training/eval stage
│ ├── ssl.yaml
│ ├── head_emb.yaml
│ ├── head_train.yaml
│ └── eval.yaml
├── scripts/ # Convenience shell scripts to launch each stage
│ ├── run_client.sh
│ ├── run_server.sh
│ └── run_eval.sh
├── data/ # Dataset location (see Data section below)
│ ├── client/
│ ├── server/
│ └── evaluation/
├── experiments/ # Code used to produce the paper's results (see below)
├── runs_E1/ … runs_E3/ # Results/figures from each paper experiment (see below)
├── pyproject.toml
├── uv.lock
└── README.md
```

| Path | What it is |
|---|---|
| `client/` | Everything that runs on the camera node: contrastive (SSL) encoder training (Stage I), embedding generation from the frozen encoder (Stage II). |
| `server/` | Everything that runs on the controller node: receives embeddings (`EmbDataset.py` wraps them for training), trains the task-specific MLP head (Stage II). |
| `models/` | Model architectures shared by both sides — encoder backbone, projection head (SimCLR/MoCo pretraining), classifier head, plus predictor networks used by the non-contrastive methods (BYOL, SimSiam). |
| `common/` | Small utility functions shared across client and server code. |
| `evaluation/` | Loads a trained head + encoder and reports test accuracy / balanced accuracy. |
| `configs/` | One YAML per stage — `ssl.yaml` (Stage I encoder training), `head_emb.yaml` (Stage II embedding generation), `head_train.yaml` (Stage II head training), `eval.yaml` (evaluation). |
| `scripts/` | Thin wrapper shell scripts for running each stage from the command line. |
| `data/` | Not tracked in git — see the Data section for how to point this at your dataset. |
| `experiments/`, `runs_E1`–`runs_E3` | Not part of the core pipeline — this is the code and output used to generate the paper's tables and figures. Documented separately below. |

