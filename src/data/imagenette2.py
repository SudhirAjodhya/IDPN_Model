from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_loaders(input_resolution: int, batch_size: int, data_root: Path, num_workers = 4):
    """
    Images are bilinear resized to a fixed dimension before batching. Use the same dimension at train and test. HFLIP is the augmentation
    """

    # Data Transforms
    training_transforms = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution), interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(data_root / "train", transform=training_transforms)
    validation_dataset = datasets.ImageFolder(data_root / "val", transform=validation_transforms)    

    # Data loaders
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))    

    return training_loader, validation_loader, len(train_dataset.classes)

