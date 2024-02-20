import torch

model = torch.jit.load('/home/mwilkers1/Documents/Projects/IMPACT-EdgeAI-Mars_/saved_models/quantized/QAT_mars_80.pt')

for name, module in model.named_modules():
    print(name, module)