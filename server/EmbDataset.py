
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

# No transforms
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return self.embeddings.size(0)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]