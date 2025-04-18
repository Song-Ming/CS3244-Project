import os
import torch
import torchvision
import torch.nn as nn
import torchinfo as info
import torchvision.transforms.v2 as v2
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

wd = os.getcwd()
device = torch.device('cuda')

# Model
model = convnext_small(weights = ConvNeXt_Small_Weights.IMAGENET1K_V1).to(device)
transform = ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()

# Replace MLP head
layers = []
layers.append(nn.Linear(in_features = 768, out_features = 256, bias = True))
layers.append(nn.GELU())
layers.append(nn.Linear(in_features = 256, out_features = 9, bias = True))
model.classifier[2] = nn.Sequential(*layers).to(device)

# Add Squeeze and Excitation modules
SqueezeExcitation1 = torchvision.ops.SqueezeExcitation(192,12,activation=nn.modules.activation.GELU)
SqueezeExcitation2 = torchvision.ops.SqueezeExcitation(384,24,activation=nn.modules.activation.GELU)
SqueezeExcitation3 = torchvision.ops.SqueezeExcitation(768,48,activation=nn.modules.activation.GELU)
model.features = nn.Sequential(model.features[0],model.features[1],model.features[2],SqueezeExcitation1,model.features[3],model.features[4],
                               SqueezeExcitation2,model.features[5],model.features[6],SqueezeExcitation3,model.features[7])

# Freeze lower layers
for i in range(3):
    for param in model.features[i].parameters():
        param.requires_grad = False
model.to(device)