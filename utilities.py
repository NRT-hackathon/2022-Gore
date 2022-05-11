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
#checkpoint =  torch.load("/home/2578/gore1/867-team-gore/lpdgen/models_done/870008_model_and_opt_save_5.torchsave")
checkpoint =  torch.load("/home/2578/gore1/867-team-gore/lpdgen/models_done/870009_model_and_opt_save_5.torchsave")

try:
    checkpoint.eval()
except AttributeError as error:
    print(error)
model.load_state_dict(checkpoint['state_dict'])

model.eval() 

batch_size_loc = 1
rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)

output = model.generator(rand_input)

print(output.shape)

##############

zzz = np.squeeze(output)
print(zzz.shape)

print(np.sum(zzz))
print(64*64*64)
zzz = np.around(zzz)
print(np.sum(zzz))

print("A")
z,x,y = zzz.nonzero()
print("B")
fig = plt.figure()
print("C")
ax = fig.add_subplot(111, projection='3d')
print("D")
ax.scatter(x, y, -z, zdir='z', c= 'grey', alpha=0.05)
print("E")
plt.show()
