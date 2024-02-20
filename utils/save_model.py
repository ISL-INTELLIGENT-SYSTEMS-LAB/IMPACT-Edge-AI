import torch
import os
from datetime import datetime

def save_model(model, torchscript=False, onnx=False, save_dir = None):
    if save_dir is None:
        torch.save(model.state_dict(), 'best_model.pt')
        return
    NOW = datetime.now().strftime("%Y-%m-%d")
    torch.save(model, os.path.join(save_dir, 'pytorch', f'PyTorch_model_{NOW}.pt'))
    if torchscript:
        model_ts = torch.jit.script(model)
        model_ts.save(os.path.join(save_dir, 'torchscript', f'TorchScript_model_{NOW}.pt'))
    if onnx:
        input_sample = torch.randn(1, 3, 224, 224)
        input_sample = input_sample.to('cuda')
        torch.onnx.export(model, input_sample, os.path.join(save_dir, 'ONNX', f'ONNX_model_{NOW}.onnx'))