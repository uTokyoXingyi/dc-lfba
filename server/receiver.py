import torch

def receive_embeddings(path: str):
    """
    currently, we just load embeddings
    """
    data = torch.load(path, map_location="cpu")
    required_keys = ["embeddings", "labels", "class_names"]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in embeddings file: {key}")

    return data