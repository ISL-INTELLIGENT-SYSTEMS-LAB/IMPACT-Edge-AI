import os
import io
import tqdm
import warnings
import torch
import torch.quantization
import torch.optim as optim
from data_preprocessing.transforms import get_train_transforms, get_val_test_transforms
from data_preprocessing.loader import create_datasets, create_data_loaders
from training.training import train_model
from utils.visualize import plot_accuracy, plot_loss
from tqdm import tqdm
from torchsummary import summary

def fuse_model(model):
    for module_name, module in model.named_children():
        if 'layer' in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, ['conv1', 'bn1', 'relu'], inplace=True)
                torch.quantization.fuse_modules(basic_block, ['conv2', 'bn2'], inplace=True)
                torch.quantization.fuse_modules(basic_block, ['conv3', 'bn3'], inplace=True)
                if hasattr(basic_block, 'downsample') and basic_block.downsample is not None:
                    torch.quantization.fuse_modules(basic_block.downsample, ['0', '1'], inplace=True)
                    print(f"Fused {module_name}.{basic_block_name}.downsample")
    return model

def setup_model_optimizer(lr_base=1e-10, lr_fc=1e-9, step_size=1, gamma=0.1, weight_decay=1e-2):
    model = torch.load('/home/mwilkers1/Documents/Projects/IMPACT-EdgeAI-Mars_/saved_models/pytorch/PyTorch_model_82_2024-01-30.pt')
    fc_params = list(model.fc.parameters())
    base_params = [param for param in model.parameters() if param.requires_grad and not any(id(param) == id(p) for p in fc_params)]
    optimizer = optim.Adam([
        {'params': base_params, 'lr': lr_base},  # Learning rate for pre-trained layers
        {'params': fc_params, 'lr': lr_fc}     # Learning rate for the fully connected layer
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader):.4f}, Accuracy: {correct / total:.4f}")
    

def evaluate(model, criterion, data_loader, device):
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(data_loader)
    test_accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return test_loss

def quant_aware_training(root_dir, data_dir, tng_labels, val_labels, test_labels, img_dir, batch_size, num_workers, checkpt_dir, epochs, patience, save_freq, save_path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")
        
        model_fp32, optimizer, scheduler, criterion = setup_model_optimizer()
        summary(model_fp32, (3, 224, 224))
                   
        model_fp32.eval()
        model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
        #activation_observer = torch.quantization.MinMaxObserver.with_args(quant_min=0, quant_max=255, dtype=torch.quint8)
        #weight_observer = torch.quantization.PerChannelMinMaxObserver.with_args(quant_min=-128, quant_max=127, dtype=torch.qint8)
        #model_fp32.qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weight_observer)
        model_fp32_fused = fuse_model(model_fp32)
        summary(model_fp32_fused, (3, 224, 224))
        model_fp32_fused.train()
        model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused, inplace=True)
        
        train_dataset, val_dataset, test_dataset = create_datasets(
            root_dir, 
            data_dir, 
            tng_labels, 
            val_labels, 
            test_labels, 
            img_dir,
            get_train_transforms(), 
            get_val_test_transforms())
        
        train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)
        #fc_params = list(model_fp32_prepared.fc.parameters())
        #base_params = [param for param in model_fp32_prepared.parameters() if param.requires_grad and not any(id(param) == id(p) for p in fc_params)]
        model_fp32_prepared.cuda()
        
        print("Performing QAT-training cycles...")
        best_loss = float('inf')
        best_model_state = None

        for epoch in range(1):
            train_one_epoch(model_fp32_prepared, criterion, optimizer, train_loader, torch.device("cuda"), epoch)
            if epoch > 70:
                model_fp32_prepared.apply(torch.quantization.disable_observer)
            if epoch > 70:
                model_fp32_prepared.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            model_fp32_prepared.cpu()
            quantized_model = torch.quantization.convert(model_fp32_prepared.eval(), inplace=False)
            quantized_model.eval()
            val_loss = evaluate(quantized_model, criterion, val_loader, torch.device("cpu"))
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model_fp32_prepared.state_dict()
                early_stopping_counter = 0

            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            

        # Save the best performing model based on loss
        # Load the best performing model state
        model_fp32_prepared.eval()
        model_fp32_prepared.load_state_dict(best_model_state)
        '''    
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model_fp32_prepared,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            torch.device("cuda"),
            checkpt_dir,
            epochs,
            patience,
            save_freq
        )
        plot_loss(train_losses, val_losses)
        plot_accuracy(train_accuracies, val_accuracies)
        '''
        #model_fp32_prepared.load_state_dict(torch.load('best_model.pt'))
        model_fp32_prepared.cpu()
        quantized_model = torch.quantization.convert(model_fp32_prepared.eval(), inplace=False)
        quantized_model.eval()
        evaluate(quantized_model, criterion, test_loader, torch.device("cpu"))
        summary(quantized_model, (3, 224, 224))
        #b = io.BytesIO()
        example_inputs = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(quantized_model, example_inputs)
        #traced_model.save(save_path)
        #torch.jit.save(torch.jit.script(model_int8), save_path)
        
        print(f"Quantized model saved to {save_path}")
        
        #return model  # Optional: return the model if you want to use it immediately