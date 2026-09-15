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
