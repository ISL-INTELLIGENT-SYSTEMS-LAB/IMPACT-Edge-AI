"""
This module contains the implementation of the MarsDataset class and functions for creating datasets and data loaders.

Classes:
- MarsDataset: A custom dataset class that inherits from the PyTorch Dataset class. It loads images and labels from files and applies transformations to the images.

Functions:
- create_datasets: Creates train, validation, and test datasets using the MarsDataset class.
- create_data_loaders: Creates train, validation, and test data loaders for the datasets.
"""

from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader

class MarsDataset(Dataset):
    """
    A PyTorch dataset class for loading and preprocessing Mars images.

    Args:
        root_dir (str): The root directory of the dataset.
        data_dir (str): The directory containing the data.
        labels_file (str): The file containing the image labels.
        img_dir (str): The directory containing the images.
        transform (callable, optional): A function/transform to apply to the images.

    Attributes:
        root_dir (str): The root directory of the dataset.
        data_dir (str): The directory containing the data.
        img_dir (str): The directory containing the images.
        transform (callable, optional): A function/transform to apply to the images.
        labels (list): The list of image labels loaded from the labels file.

    Methods:
        _load_labels(labels_file): Load the labels from the labels file.
        _check_images_exist(): Check that all images actually exist.
        __len__(): Return the total number of samples.
        __getitem__(idx): Return the image and its label.
    """
    
    def __init__(self, root_dir, data_dir, labels_file, img_dir, transform=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self._load_labels(labels_file)
        self._check_images_exist()

    def _load_labels(self, labels_file):
        labels_path = os.path.join(self.root_dir, self.data_dir, labels_file)
        with open(labels_path, 'r') as file:
            labels = [tuple(line.strip().split()) for line in file]
        return labels

    def _check_images_exist(self):
        for img_name, _ in self.labels:
            img_path = os.path.join(self.root_dir, self.data_dir, self.img_dir, img_name)
            assert os.path.isfile(img_path), f"File not found: {img_path}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, image_label = self.labels[idx]
        image_path = os.path.join(self.root_dir, self.data_dir, self.img_dir, img_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, int(image_label)
    
def create_datasets(root_dir, data_dir, train_labels, val_labels, test_labels, img_dir, train_transform, val_test_transform):
    """
    Create datasets for training, validation, and testing.

    Parameters:
    root_dir (str): Root directory of the dataset.
    data_dir (str): Directory containing the dataset.
    labels_file (str): File containing the labels.
    img_dir (str): Directory containing the images.
    train_transform (callable): Transformation to apply to training data.
    val_test_transform (callable): Transformation to apply to validation and testing data.

    Returns:
    train_dataset (MarsDataset): Dataset for training.
    val_dataset (MarsDataset): Dataset for validation.
    test_dataset (MarsDataset): Dataset for testing.
    """
    train_dataset = MarsDataset(root_dir, data_dir, train_labels, img_dir, transform=train_transform)
    val_dataset = MarsDataset(root_dir, data_dir, val_labels, img_dir, transform=val_test_transform)
    test_dataset = MarsDataset(root_dir, data_dir, test_labels, img_dir, transform=val_test_transform)
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=16):
    """
    Create data loaders for training, validation, and testing datasets.

    Parameters:
    train_dataset (Dataset): The training dataset.
    val_dataset (Dataset): The validation dataset.
    test_dataset (Dataset): The testing dataset.
    batch_size (int): The batch size for the data loaders. Default is 64.
    num_workers (int): The number of worker threads for data loading. Default is 16.

    Returns:
    train_loader (DataLoader): The data loader for the training dataset.
    val_loader (DataLoader): The data loader for the validation dataset.
    test_loader (DataLoader): The data loader for the testing dataset.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader