import os
import numpy as np
from  models import DCGAN3D
import time
import h5py
from iolocal import save_hdf5
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning
import sys
import imageio

###  Generate several outputs for a saved model
#
if len(sys.argv) != 4:
     print('Call this program using python3 ' + sys.argv[0] + ' [number as seed] [model checkpoint] [output file]')
     sys.exit()

ts_start = time.time()
print("Generation of volumes")


a_seed = int(sys.argv[1])       # Number of samples we make from the model.
# Where to load model from
checkpoint_name = sys.argv[2]

# Where to save the generated volumes (the artificial materials)
file_path_generated_volume = sys.argv[3]


model = DCGAN3D()
checkpoint =  torch.load(checkpoint_name)
model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
model.eval() # ensure we are in evaluation (not training mode)

torch.manual_seed(a_seed)
# latent dim may depend on the model
latent_dim = 64 # hardcoded for now
hdf_keyname = 'data'

rand_unit1 = torch.randn(1 * latent_dim).view(1, latent_dim, 1, 1, 1)
rand_unit1 /= torch.linalg.vector_norm(rand_unit1)
rand_unit2 = torch.randn(1 * latent_dim).view(1, latent_dim, 1, 1, 1)
rand_unit2 /= torch.linalg.vector_norm(rand_unit2)

num_interpolations = 7
sigma_scales = np.sqrt(latent_dim)*np.linspace(-1,1,num_interpolations)
#sigma_scales = np.linspace(-3,3,num_interpolations)

a_array = []
b_array = []
for i in range(num_interpolations):
  a = []
  b = []
  for j in range(num_interpolations):
    rand_input = rand_unit1*sigma_scales[i] + rand_unit2*sigma_scales[j]
    output = model.generator(rand_input).detach()
    output = torch.squeeze(output)
    bin_output = np.around(output.numpy())
    a_slice = np.mean(bin_output,axis=0)
    b_slice = bin_output[31]
    a_slice = np.append(np.append(a_slice,0.5*np.ones( (64,1)),axis=1), 0.5*np.ones((1,65)),axis=0)
    b_slice = np.append(np.append(b_slice,0.5*np.ones( (64,1)),axis=1), 0.5*np.ones((1,65)),axis=0)
    a.append(a_slice)
    b.append(b_slice)
  a_array.append(a)
  b_array.append(b)

A = np.block(a_array)
B = np.block(b_array)


a_path = os.path.join(file_path_generated_volume,  'gen_' + str(a_seed))
file = a_path+'_aved_slices' + '.png'
imageio.imwrite(file, A)
file = a_path+'_bin_slice' + '.png'
imageio.imwrite(file, B)


total_time = time.time() - ts_start

print("Generation of volumes done.")
print("Wall time expended: " + str(total_time) + " seconds.")
print("Generated volumes are in " + file_path_generated_volume)
print("Now run analysis with GooseEYE installed.")

### END: Generate an output for a saved model #######################################################################################

