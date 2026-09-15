import os
from typing import Any, Dict, List

from PIL import Image #used for images open and saving
import torchvision.transforms as T
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    def __init__(self, root: str, image_size: int=224, method: str = "simclr"):
        self.root = root   # our dataset is root="data/client/unlabeled"
        self.image_size = image_size # output size
        self.method = method.lower()

        self.image_paths = self._load_image_paths(root) # used to get the path for each image
        self.transform = self._build_transform()

    def _load_image_paths(self, root: str) -> List[str]: # get the path of each image
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        paths = []
        for dirpath, _, filenames in os.walk(root):  # os.walk() returns (dirpath, dirnames, filenames), each one is a list
            for fname in filenames:
                if fname.lower().endswith(image_extensions):
                    paths.append(os.path.join(dirpath, fname))  # joint(dir,name) ===> unlabeled/0001.jpg ; unlabeled/0002.jpg
        
        if len(paths)==0:
            raise RuntimeError(f"No images found in {root}")
        return sorted(paths)
    
    def _build_transform(self):
        if self.method == "simclr":
            print("dataset for simclr")
            return T.Compose([  # combine multiple transformation into a composer.
                    T.RandomResizedCrop(self.image_size, scale=(0.2, 1.0)), #ariations in scale and aspect ratio
                    # T.Resize((self.image_size, self.image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([
                        T.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                            hue=0.1
                        )
                    ], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.GaussianBlur(int(0.1 * self.image_size) // 2 * 2 + 1, sigma=(0.1, 2.0)), # Jau. 29, 2026
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    ),
                ])
        elif self.method == "moco":
            print("dataset for moco")
            return T.Compose([
                # T.RandomResizedCrop(self.image_size, scale=(0.4, 1.0)),
                T.RandomResizedCrop(self.image_size, scale=(0.6, 1.0)),

                T.RandomHorizontalFlip(p=0.5),

                # T.ColorJitter(0.3, 0.3, 0.3, 0.05),
                T.ColorJitter(0.15, 0.15, 0.15, 0.03),

                # T.GaussianBlur(
                #     kernel_size=int(0.1 * self.image_size) // 2 * 2 + 1,
                #     sigma=(0.1, 1.0)
                # ),

                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        elif self.method == "byol":
            print("dataset for byol")
            return T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.6, 1.0)),
                
                T.RandomHorizontalFlip(p=0.5),

                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                # T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)],0.8),

                T.GaussianBlur(
                    kernel_size=int(0.1 * self.image_size) // 2 * 2 + 1,
                    sigma=(0.1, 1.0)
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        elif self.method == "simsiam":
            print("dataset for simsiam")
            return T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(p=0.5),

                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                # T.GaussianBlur(
                #     kernel_size=int(0.1 * self.image_size) // 2 * 2 + 1,
                #     sigma=(0.1, 1.0)
                # ),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])

    def __len__(self) -> int:
        return len(self.image_paths) #return the number of images
    def __getitem__(self, idx: int): # used to load data
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        x_i = self.transform(image)
        x_j = self.transform(image)
        return x_i, x_j
