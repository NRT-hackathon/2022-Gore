import argparse
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning

from iolocal import *
import os
import numpy as np
from sklearn.ensemble._hist_gradient_boosting import loss
from models import DCGAN3D
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
     print('Call this program using python3 ' + sys.argv[0] + ' <run_number> <epoch_number>')
     sys.exit()

width = 20
height = 20
alpha = 0.1

run_num = sys.argv[1]
epoch_num = sys.argv[2]

print("Opening /home/2578/gore1/867-team-gore/lpdgen/models_done/" + str(run_num) + "_model_and_opt_save_" + str(epoch_num) + ".torchsave")

# https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
model = DCGAN3D()
checkpoint =  torch.load("/home/2578/gore1/867-team-gore/lpdgen/models_done/" + str(run_num) + "_model_and_opt_save_" + str(epoch_num) + ".torchsave")

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

file_name = "3d_" + str(run_num) + "_" + str(epoch_num) + '.npy'

np.save(file_name, output)
print("saved to: ./" + file_name)

bb = np.load(file_name)

print(bb.shape)

zzz = np.squeeze(bb)
print(zzz.shape)

print(np.sum(zzz))
print(64*64*64)
zzz = np.around(zzz)
print(np.sum(zzz))

print("Beginning creation of image")
z,x,y = zzz.nonzero()
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'grey', alpha=alpha)
print("Image ready to save")
#plt.show()
plt.savefig(file_name + '.png')
print("Saved to: " + file_name + '.png')
