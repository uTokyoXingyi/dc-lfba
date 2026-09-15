# client_V0: the first idea -- DC-LFBA
- ssl_trainer_SimCLR: only related to SimCLR
- ssl_trainer: Combine MoCo, SimSaim, BYOL, SimCLR together
## parameters
training:
  epochs: 100
  batch_size: 32
  num_workers: 8
  learning_rate: 0.01
  weight_decay: 1e-4
  temperature: 0.5
  seed: 42
optimizer:
  type: "sgd"
  momentum: 0.9

data:
  root: "data/client/unlabeled"
  image_size: 160 #output size

checkpoint:
  save_dir: "checkpoints/ssl"
  save_interval: 10   # epochs
  ssl_csv_path: "checkpoints/ssl_metrics.csv"

model:
  projection_dim: 128

logging:
  log_interval: 100

## training
SimCLR
batch size 32 -- image size 224
batch size 64 -- image size 160