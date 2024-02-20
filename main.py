import os
import sys
import torch
from data_preprocessing.transforms import get_train_transforms, get_val_test_transforms
from data_preprocessing.loader import create_datasets, create_data_loaders
from models.resnet50 import setup_model_optimizer
from training.training import train_model, evaluate_model
from utils.visualize import plot_accuracy, plot_loss, plot_confusion_matrix, plot_normalized_confusion_matrix
from utils.save_model import save_model
from utils.post_training_quantization import post_training_quantization
from utils.quant_aware_training import quant_aware_training
import datetime

def train_nonquantized():
    NOW = datetime.datetime.now()
    ROOT_DIR = os.getcwd()
    DATA_DIR = 'data'
    TNG_LABELS = os.path.join('labels', 'train-set-v2.1.txt')
    VAL_LABELS = os.path.join('labels', 'val-set-v2.1.txt')
    TEST_LABELS = os.path.join('labels', 'test-set-v2.1.txt')
    IMG_DIR = 'mars_images'
    QUANT_SAVE_PATH = os.path.join(ROOT_DIR, 'saved_models', 'quantized', f'{NOW}+quantized_model')
    CLASS_NAMES = ['arm cover', 'other rover part', 'artifact', 'nearby surface', 'close-up rock', 
                'DRT','DRT spot', 'distant landscape', 'drill hole', 'night sky', 'light-toned veins', 
                'mastcam cal target','sand', 'sun', 'wheel', 'wheel joint', 'wheel tracks']
    CHECKPT_DIR = os.path.join(ROOT_DIR, 'saved_models', 'checkpoints')
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    EPOCHS = 200
    PATIENCE = 3
    SAVE_FREQUENCY = 5
    LR_BASE = 1e-6
    LR_FC = 1e-5
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 4
    
    train_dataset, val_dataset, test_dataset = create_datasets(
        ROOT_DIR, 
        DATA_DIR, 
        TNG_LABELS, 
        VAL_LABELS, 
        TEST_LABELS, 
        IMG_DIR,
        get_train_transforms(), 
        get_val_test_transforms())
    
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS)
    model, optimizer, scheduler, criterion = setup_model_optimizer(LR_BASE, LR_FC, WEIGHT_DECAY, STEP_SIZE)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        CHECKPT_DIR,
        EPOCHS,
        PATIENCE,
        SAVE_FREQUENCY
    )
    
    plot_loss(train_losses, val_losses)
    plot_accuracy(train_accuracies, val_accuracies)

    conf_matrix, conf_matrix_normalized = evaluate_model(model, test_loader, criterion, device)
    plot_confusion_matrix(conf_matrix, CLASS_NAMES)
    plot_normalized_confusion_matrix(conf_matrix_normalized, CLASS_NAMES)

    save_model(model, torchscript=True, onnx=True, save_dir = os.path.join(ROOT_DIR, 'saved_models'))

def QAT():
    NOW = datetime.datetime.now()
    ROOT_DIR = os.getcwd()
    DATA_DIR = 'data'
    TNG_LABELS = os.path.join('labels', 'train-set-v2.1.txt')
    VAL_LABELS = os.path.join('labels', 'val-set-v2.1.txt')
    TEST_LABELS = os.path.join('labels', 'test-set-v2.1.txt')
    IMG_DIR = 'mars_images'
    QUANT_SAVE_PATH = os.path.join(ROOT_DIR, 'saved_models', 'quantized', f'{NOW}+quantized_model.pt')
    CHECKPT_DIR = os.path.join(ROOT_DIR, 'saved_models', 'checkpoints')
    BATCH_SIZE = 8
    NUM_WORKERS = 16
    EPOCHS = 20
    PATIENCE = 10
    SAVE_FREQUENCY = 5
    quant_aware_training(ROOT_DIR, DATA_DIR, TNG_LABELS, VAL_LABELS, TEST_LABELS, IMG_DIR, BATCH_SIZE, NUM_WORKERS, CHECKPT_DIR, EPOCHS, PATIENCE, SAVE_FREQUENCY, QUANT_SAVE_PATH)

def PTQ(model_path: str):
    NOW = datetime.datetime.now()
    ROOT_DIR = os.getcwd()
    DATA_DIR = 'data'
    TNG_LABELS = os.path.join('labels', 'train-set-v2.1.txt')
    VAL_LABELS = os.path.join('labels', 'val-set-v2.1.txt')
    TEST_LABELS = os.path.join('labels', 'test-set-v2.1.txt')
    IMG_DIR = 'mars_images'
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    QUANT_SAVE_PATH = os.path.join(ROOT_DIR, 'saved_models', 'quantized', f'{NOW}+quantized_model.pt')
    post_training_quantization(ROOT_DIR, DATA_DIR, TNG_LABELS, VAL_LABELS, TEST_LABELS, IMG_DIR, BATCH_SIZE, NUM_WORKERS, QUANT_SAVE_PATH, model_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'train':  # Train the model (non-quantized)
            print("Training model (non-quantized)...")
            train_nonquantized()
        elif command == 'PTQ':  # PTQ: Post Training Quantization
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
                print(f"Quantizing model (PTQ) with model at {model_path}...")
                PTQ(model_path)
            else:
                print("Please provide the path to the trained model for PTQ.")
        elif command == 'QAT':  # QAT: Quantization Aware Training
            print(f"Quantizing model (QAT)...")
            QAT()
        
    else:
        print("Please provide an argument to execute the code.")
