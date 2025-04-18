import os
import torch
import torchvision
import torch.nn as nn
import torchinfo as info
import torchvision.transforms.v2 as v2
from torchvision.models import swin_s, Swin_S_Weights

wd = os.getcwd()
device = torch.device('cuda')

# Model
model = swin_s(weights = Swin_S_Weights.IMAGENET1K_V1).to(device)
transform = Swin_S_Weights.IMAGENET1K_V1.transforms()

# Replace MLP head
layers = []
layers.append(nn.Linear(in_features = 768, out_features = 256, bias = True))
layers.append(nn.GELU())
layers.append(nn.Linear(in_features = 256, out_features = 9, bias = True))
model.head = nn.Sequential(*layers).to(device)

# Freeze lower layers
for i in range(3):
    for param in model.features[i].parameters():
        param.requires_grad = False
model.to(device)
