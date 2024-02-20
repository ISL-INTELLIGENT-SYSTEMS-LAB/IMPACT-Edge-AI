import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models

def pyquant_test_cpu(model_path, img_dir):
    model = torch.jit.load(model_path)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cpu")
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        img = Image.open(img_path)
        img = preprocess(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
        print(output)
        del img
        del output

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(root_dir, 'saved_models', 'quantized', '2024-01-31_15:34:20.974927+quantized_model.pt')
    data_dir = os.path.join(root_dir, 'data')
    img_dir = os.path.join(data_dir, 'mars_images')
    pyquant_test_cpu(model_path, img_dir)