

"""
This module contains the function `kfold_cross_validation` which performs k-fold cross-validation for training a machine learning model.

The `kfold_cross_validation` function takes the following parameters:
- train_dataset: The training dataset.
- val_dataset: The validation dataset.
- model: The machine learning model to be trained.
- optimizer: The optimizer used for training the model.
- scheduler: The learning rate scheduler.
- criterion: The loss function.
- k: The number of folds for cross-validation (default is 5).
- batch_size: The batch size for training (default is 64).
- num_workers: The number of worker threads for data loading (default is 16).

The function performs k-fold cross-validation by splitting the combined train and validation datasets into k folds. For each fold, it trains the model using the training data and evaluates it on the validation data. It keeps track of the training and validation losses and accuracies for each fold. After all folds are processed, it calculates the average training and validation losses and accuracies, as well as the variability between folds.

The function also plots the training and validation losses and accuracies for each fold.

Example usage:
train_dataset = ...
val_dataset = ...
model = ...
optimizer = ...
scheduler = ...
criterion = ...
kfold_cross_validation(train_dataset, val_dataset, model, optimizer, scheduler, criterion)
"""

import torch
import numpy as np
import matplotlib as plt
from sklearn.model_selection import KFold
from training import train_model

def kfold_cross_validation(train_dataset, val_dataset, model, optimizer, scheduler, criterion, k=5, batch_size=64, num_workers=16):
    # Concatenate train_dataset and val_dataset
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    train_dataset = combined_dataset

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True)
    fold = 1

    # Initialize lists to store metrics for each fold
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    # Initialize variables to store the best model  
    best_val_accuracy = 0.0
    best_model = None
        
    for train_index, val_index in kf.split(train_dataset):

        print(f"Training on Fold {fold}")
        
        # Create train and validation data loaders for the current fold
        train_fold_dataset = torch.utils.data.Subset(train_dataset, train_index)
        val_fold_dataset = torch.utils.data.Subset(train_dataset, val_index)
        train_loader = torch.utils.data.DataLoader(train_fold_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_fold_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        model.to(device) # Move model to GPU if available

        epochs = 10  # Number of epochs to train the model
        patience = 3  # Number of epochs to wait for improvement in validation loss before early stopping
        save_frequency = 5# Frequency (in epochs) at which to save model checkpoints
            
        # Train the model
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            patience=patience,
            save_frequency=save_frequency
            )
        # Append the metrics for the current fold to the lists    
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)

        # Increment the fold number    
        fold += 1  

        # Save the best model based on validation accuracy
        current_val_accuracy = max(val_accuracies)
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            #best_model_state = model.state_dict()
            #print("Best model updated.")
                        
        # Reinitialize the model for the next fold
        model = None
        
    # Calculate average training and validation loss
    avg_train_loss = np.mean([min(fold_losses) for fold_losses in all_train_losses])
    avg_val_loss = np.mean([min(fold_losses) for fold_losses in all_val_losses])
      
    # Calculate average training and validation accuracy
    avg_train_accuracy = np.mean([max(fold_accuracies) for fold_accuracies in all_train_accuracies])
    avg_val_accuracy = np.mean([max(fold_accuracies) for fold_accuracies in all_val_accuracies])
        
    # Calculate variability between folds
    val_accuracy_std = np.std([max(fold_accuracies) for fold_accuracies in all_val_accuracies])
        
    print("Average Training Loss:", avg_train_loss)
    print("Average Validation Loss:", avg_val_loss)
    print("Average Training Accuracy:", avg_train_accuracy)
    print("Average Validation Accuracy:", avg_val_accuracy)
    print("Variability between Folds (Validation Accuracy):", val_accuracy_std)

    # Plotting all training losses
    plt.figure(figsize=(10, 5))
    for i, fold_losses in enumerate(all_train_losses):
        plt.plot(range(1, len(fold_losses) + 1), fold_losses, label=f'Fold {i+1}')
    plt.title('Training Loss per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting all validation losses
    plt.figure(figsize=(10, 5))
    for i, fold_losses in enumerate(all_val_losses):
        plt.plot(range(1, len(fold_losses) + 1), fold_losses, label=f'Fold {i+1}')
    plt.title('Validation Loss per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting all training accuracies
    plt.figure(figsize=(10, 5))
    for i, fold_accuracies in enumerate(all_train_accuracies):
        plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, label=f'Fold {i+1}')
    plt.title('Training Accuracy per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting all validation accuracies
    plt.figure(figsize=(10, 5))
    for i, fold_accuracies in enumerate(all_val_accuracies):
        plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, label=f'Fold {i+1}')
    plt.title('Validation Accuracy per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
