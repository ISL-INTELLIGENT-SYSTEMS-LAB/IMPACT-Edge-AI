import torch
from onnxruntime.quantization import quantize_dynamic
import os
import onnx

def convert_and_quantize_model(pytorch_model, quantized_model_path):
    pytorch_model.eval()  # Set the model to inference mode
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input shape
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pytorch_model.to(device)
    dummy_input = dummy_input.to(device)
    onnx_model_path = "converted_model.onnx"
    
    # Export the model to ONNX
    torch.onnx.export(pytorch_model, dummy_input, onnx_model_path)
    
    # Quantize the ONNX model
    quantize_dynamic(onnx_model_path, quantized_model_path)
    os.remove(onnx_model_path)
    
    size_in_bytes = os.path.getsize(quantized_model_path)
    size_in_mb = size_in_bytes / (1024 * 1024)
    
    print(f"Quantized model saved to {quantized_model_path}")
    print(f"Size of quantized model: {size_in_mb:.2f} MB")

    onnx_model = onnx.load(quantized_model_path)

