"""
#import argparse
#import torch.nn as nn
#import torch.nn.parallel
#import pytorch_lightning as lightning
"""

#from iolocal import *

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.ensemble._hist_gradient_boosting import loss
#from models import DCGAN3D

"""
fl = "./869681_generated_prediction_0.torchtensor"

voxels = torch.load(fl)

voxels_np = voxels.cpu().detach().numpy()
voxels_np = np.asarray(voxels_np)

z,x,y = voxels_np.nonzero()
print("B")
fig = plt.figure()
print("C")
ax = fig.add_subplot(111, projection='3d')
print("D")
ax.scatter(x, y, -z, zdir='z', c= 'grey')
print("E")
plt.savefig("demo.png")
    
print("done")
"""

### START: Generate an output for a saved model #######################################################################################

"""
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

#output = output.cpu().numpy()
output = output.detach().numpy()

print("----")
print(output.shape)

np.save('3d.npy', output)

"""

#run_num = '876551'

#run_num = '879156'
#epoch_num = '13'

#run_num = '880451'
#epoch_num = '22'

#run_num = '880696'
#epoch_num = '14'

#run_num = '880890'
#epoch_num = '39'

#run_num = '881176'
#epoch_num = '51'

#run_num = '881309'
#epoch_num = '6'

#run_num = '881507'
#epoch_num = '58'

#run_num = '881508'
#epoch_num = '11' 

#run_num = '882670'
#epoch_num = '73' 

#run_num = '882672'
#epoch_num = '24' 

#run_num = '883190'
#epoch_num = '35' 

#run_num = '883191'
#epoch_num = '84' 

#run_num = '883366'
#epoch_num = '22' 

#run_num = '883429'
#epoch_num = '102' 

#run_num = '883430'
#epoch_num = '53' 

################

#run_num = '889158'
#epoch_num = '31' 

#run_num = '889159'
#epoch_num = '83' 

#run_num = '889160'
#epoch_num = '133' 

#run_num = '889309'
#epoch_num = '7'

################

#run_num = '889543'
#epoch_num = '95'

#run_num = '889562'
#epoch_num = '25'

run_num = '889570'
epoch_num = '44'

#run_num = '889521'
#epoch_num = '143'


#alpha_const = 0.7
alpha_const = 0.05


round_to_one_and_zero = True
#round_to_one_and_zero = False

file_path = 'C:\\!data\\college\\2022 Spring\\CISC867 - Soft Materials\\goreCode\\working\\predictions\\imagesAndNpy\\'

file_name = file_path + run_num + '_generatedvolume_' + epoch_num+ '.npy'

bb = np.load(file_name)

print(bb.shape)

zzz = np.squeeze(bb)
print(zzz.shape)

print(np.sum(zzz))
print(64*64*64)
#zzz = np.around(zzz)

if round_to_one_and_zero:
    # Force to exactly zero and exactly one.
    print("Force to exactly zero and exactly one.")
    zzz = np.around(zzz)
    print(np.sum(zzz))
    z,x,y = zzz.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'grey', alpha=alpha_const)
else:
    print(np.sum(zzz))
   
    x = zzz[:, 0:1]
    y = zzz[:, 1:2]
    z = zzz[:, 2:3]   
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'grey', alpha=alpha_const)
   

print("ready")
plt.show()
#plt.savefig("demo.png")

### END: Generate an output for a saved model #######################################################################################

