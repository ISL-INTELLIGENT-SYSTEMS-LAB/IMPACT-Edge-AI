import numpy as np
from sklearn.utils import class_weight
import torch
from torchvision import models
import torchvision.models.quantization
import torch.optim as optim

def setup_model_optimizer(lr_base=1e-10, lr_fc=1e-9, step_size=2, gamma=0.1, weight_decay=1e-2):
    """
    Set up the ResNet-50 model, optimizer, scheduler, and criterion.

    Returns:
        model (torchvision.models.ResNet): The ResNet-50 model with the last fully connected layer modified for the specified number of classes.
        optimizer (torch.optim.Adam): The optimizer used for training the model.
        scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler used for adjusting the learning rate during training.
        criterion (torch.nn.CrossEntropyLoss): The loss function used for calculating the loss during training.
    """
    model = models.quantization.resnet50(weights=models.ResNet50_Weights.DEFAULT, quantize = False)
    num_features = model.fc.in_features
    num_classes = 19
    model.fc = torch.nn.Linear(num_features, num_classes)
    fc_params = list(model.fc.parameters())
    base_params = [param for param in model.parameters() if param.requires_grad and not any(id(param) == id(p) for p in fc_params)]
    optimizer = optim.Adam([
        {'params': base_params, 'lr': lr_base},  # Learning rate for pre-trained layers
        {'params': fc_params, 'lr': lr_fc}     # Learning rate for the fully connected layer
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion