import os
import io
import warnings
import torch
import torch.quantization
from data_preprocessing.transforms import get_train_transforms, get_val_test_transforms
from data_preprocessing.loader import create_datasets, create_data_loaders

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

def calibrate(model, dataloader):
    model.eval()
    model.cpu()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to('cpu')
            model(data)
    return model

def post_training_quantization(root_dir, data_dir, tng_labels, val_labels, test_labels, img_dir, batch_size, num_workers, save_path: str, model_path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")
        
        model_fp32 = torch.load(model_path)
        model_fp32.eval()
        model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
        model_fp32.qconfig_dict = {'weight': torch.quantization.default_per_channel_qconfig, # this did not work well as defaults, remove activation observer
                                   'activation': torch.quantization.default_observer}
        model_fp32_fused = fuse_model(model_fp32)
        model_fp32_prepared = torch.quantization.prepare(model_fp32_fused, inplace=True)
        # refactor to only use val_dataset
        train_dataset, val_dataset, test_dataset = create_datasets(
            root_dir, 
            data_dir, 
            tng_labels, 
            val_labels, 
            test_labels, 
            img_dir,
            get_train_transforms(), 
            get_val_test_transforms())
        # refactor to only use val_loader
        train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)
        print("Calibrating model...")
        model_fp32_prepared = calibrate(model_fp32_prepared, train_loader)
        model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=True)
        #b = io.BytesIO()
        example_inputs = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(model_int8, example_inputs)
        traced_model.save(save_path)
        #torch.jit.save(torch.jit.script(model_int8), save_path)
        print(f"Quantized model saved to {save_path}")

        #return model  # Optional: return the model if you want to use it immediately