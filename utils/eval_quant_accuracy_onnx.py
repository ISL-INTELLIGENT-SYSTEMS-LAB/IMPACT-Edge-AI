import os
import sys
import numpy as np
import torch
import onnxruntime
from torchvision import transforms
from data_preprocessing.loader import create_datasets, create_data_loaders
from data_preprocessing.transforms import get_train_transforms, get_val_test_transforms

def main(model_path):
    # Assuming ROOT_DIR and other constants are defined outside this snippet
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

    _, _, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS)

    session = onnxruntime.InferenceSession(model_path)
    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Convert inputs to the format expected by ONNX
            inputs_np = inputs.numpy()
            inputs_dict = {session.get_inputs()[0].name: inputs_np}

            # Run inference
            ort_outs = session.run(None, inputs_dict)
            outputs = torch.from_numpy(ort_outs[0]).float()  # Ensure outputs are floats

            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Finalize and print results
    test_loss /= len(test_loader)
    test_accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the model file as a command line argument.")
        sys.exit(1)
    main(sys.argv[1])
