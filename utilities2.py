import argparse
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning
from iolocal import *
import os
import numpy as np
from sklearn.ensemble._hist_gradient_boosting import loss
from models import DCGAN3D

# https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
model = DCGAN3D()
checkpoint =  torch.load("/home/2578/gore1/867-team-gore/lpdgen/models_done/870008_model_and_opt_save_5.torchsave")
try:
    checkpoint.eval()
except AttributeError as error:
    print(error)

for key in checkpoint:
    print(key)


model.load_state_dict(checkpoint['model_state_dict'])
model.eval() 

batch_size_loc = 1
rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)

output = model.generator(rand_input)

print(output.shape)

output = output.detach().numpy()

print("----")
print(output.shape)

np.save('3d.npy', output)
