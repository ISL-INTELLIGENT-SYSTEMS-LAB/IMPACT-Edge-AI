

"""
This module contains functions for training and evaluating a model using PyTorch.

Functions:
- train_model: Trains a model using the provided data loaders, criterion, optimizer, and scheduler.
- evaluate_model: Evaluates the performance of a model on a test dataset.

"""

import torch
import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.save_model import save_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_path, epochs=10, patience=3, save_frequency=5):
    """
    Trains a model using the provided data loaders, criterion, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to be used for training.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 3.
        save_frequency (int, optional): The frequency (in epochs) at which to save checkpoints. Defaults to 5.

    Returns:
        tuple: A tuple containing lists of training losses, validation losses, training accuracies, and validation accuracies.
    """
    
    def forward_pass(model, inputs, labels, criterion, device): 
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        return outputs, loss, predicted
    
    def backward_pass(optimizer, loss): 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def compute_accuracy(predicted, labels, device):
        labels = labels.to(device)
        predicted = predicted.to(device)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct, total

    def train_one_epoch(model, train_loader, criterion, optimizer, device): 
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm.tqdm(train_loader, leave=False):
            outputs, loss, predicted = forward_pass(model, inputs, labels, criterion, device)
            if torch.isnan(loss):
                print("Loss is NaN. Stopping training.")
                return None, None
            backward_pass(optimizer, loss)
            running_loss += loss.item()
            c, t = compute_accuracy(predicted, labels, device)
            correct += c
            total += t
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        return train_loss, train_accuracy

    def validate_one_epoch(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs, loss, predicted = forward_pass(model, inputs, labels, criterion, device)
                val_loss += loss.item()
                c, t = compute_accuracy(predicted, labels, device)  # Pass the 'device' argument to compute_accuracy
                correct += c
                total += t
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy
        
    def print_epoch_stats(epoch, epochs, train_loss, val_loss, train_accuracy, val_accuracy):
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    def save_checkpoint(epoch, model, optimizer, save_path='checkpoint.pth'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
         }, save_path)

    best_val_loss = float('inf')
    trigger_times = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f'Training Epoch {epoch+1}/{epochs}')
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if train_loss is None:  # NaN Loss Check
            break
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            save_model(model, torchscript=True, onnx=True)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        scheduler.step()
        print_epoch_stats(epoch, epochs, train_loss, val_loss, train_accuracy, val_accuracy)
        #if save_path is not None:
            #save_path = os.path.join(save_path, f'checkpoint_{epoch+1}_loss_{val_loss}_accuracy_{val_accuracy}.pth')
        #if epoch % save_frequency == 0:
            #save_checkpoint(epoch, model, optimizer, save_path)

    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device on which the evaluation will be performed.

    Returns:
        cm (numpy.ndarray): The confusion matrix.
        cm_normalized (numpy.ndarray): The normalized confusion matrix.
    """
    def forward_pass(inputs, labels, model, criterion, device):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return outputs, loss.item()

    def update_metrics(outputs, labels, loss, correct, total, all_predicted, all_labels):
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.to(device)).sum().item()
        total += labels.size(0)
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        return loss, correct, total

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, batch_loss = forward_pass(inputs, labels, model, criterion, device)
            test_loss += batch_loss
            test_loss, correct, total = update_metrics(outputs, labels, test_loss, correct, total, all_predicted, all_labels)

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    cm = confusion_matrix(all_labels, all_predicted)
    epsilon = 1e-7  # small constant
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + epsilon)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    return cm, cm_normalized