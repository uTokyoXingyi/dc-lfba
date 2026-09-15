python3 -m client.client --ssl-config configs/ssl.yaml --emb-config configs/head.yaml

python3 test_embeddings.py --ssl-config configs/ssl.yaml --emb-config configs/head_emb.yaml # only for embedding computation

python3 -m client.client --ssl-config configs/ssl.yaml --emb-config configs/head_emb.yaml

torchrun --nproc_per_node=2 -m client.client --ssl-config configs/ssl.yaml --emb-config configs/head_emb.yaml