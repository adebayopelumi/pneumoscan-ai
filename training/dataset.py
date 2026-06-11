from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from training.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(augment=True))
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_transforms())
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.classes
