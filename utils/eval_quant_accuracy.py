import os
import sys
import torch
import numpy as np
from torchvision import transforms
from ..data_preprocessing.loader import create_datasets, create_data_loaders
from ..data_preprocessing.transforms import get_train_transforms, get_val_test_transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = 'data'
TNG_LABELS = os.path.join('labels', 'train-set-v2.1.txt')
VAL_LABELS = os.path.join('labels', 'val-set-v2.1.txt')
TEST_LABELS = os.path.join('labels', 'test-set-v2.1.txt')
IMG_DIR = 'mars_images'
CLASS_NAMES = ['arm cover', 'other rover part', 'artifact', 'nearby surface', 'close-up rock', 
               'DRT','DRT spot', 'distant landscape', 'drill hole', 'night sky', 'light-toned veins', 
               'mastcam cal target','sand', 'sun', 'wheel', 'wheel joint', 'wheel tracks']
BATCH_SIZE = 64
NUM_WORKERS = 16

def main(model_path):
    train_dataset, val_dataset, test_dataset = create_datasets(
        ROOT_DIR, 
        DATA_DIR, 
        TNG_LABELS, 
        VAL_LABELS, 
        TEST_LABELS, 
        IMG_DIR,
        get_train_transforms(), 
        get_val_test_transforms()
    )

    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS)

    model = torch.jit.load(model_path)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')  # Separate the print statement from the previous code block with a newline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the model file as a command line argument.")
        sys.exit(1)
    MODEL_PATH = sys.argv[1]
    main(MODEL_PATH)
