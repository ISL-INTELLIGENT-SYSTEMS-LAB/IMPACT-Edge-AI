

from torchvision import transforms

"""
This module provides functions to obtain composition of transformations for data preprocessing.

The module includes the following functions:
- get_train_transforms: Returns a composition of transformations to be applied to the training images.
- get_val_test_transforms: Returns a composition of transformations to be applied to the validation and test images.
"""

def get_train_transforms():
    """
    Returns a composition of transformations to be applied to the training images.

    Returns:
        transforms.Compose: A composition of transformations.
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.CenterCrop(size=224),
        transforms.RandomGrayscale(p=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.57749283, 0.46066804, 0.37339427], std=[0.10846638, 0.09689421, 0.09173507]), 
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

def get_val_test_transforms():
    """
    Returns a composition of transformations to be applied to the validation and test images.

    Returns:
        transforms.Compose: A composition of transformations.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.57749283, 0.46066804, 0.37339427], std=[0.10846638, 0.09689421, 0.09173507]), 
    ])