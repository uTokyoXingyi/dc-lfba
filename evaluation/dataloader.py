# load test data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_test_loader( data_root, image_size, batch_size, num_workers):
    test_transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.Resize((image_size, image_size)),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    test_dataset = datasets.ImageFolder(
        root=str(data_root),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_dataset, test_loader