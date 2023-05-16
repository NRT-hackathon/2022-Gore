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
     print('Call this program using python3 ' + sys.argv[0] + ' [number to generate] [model checkpoint] [output file]')
     sys.exit()

ts_start = time.time()
print("Generation of volumes")


num_samples = int(sys.argv[1])       # Number of samples we make from the model.
# Where to load model from
checkpoint_name = sys.argv[2]

# Where to save the generated volumes (the artificial materials)
file_path_generated_volume = sys.argv[3]


model = DCGAN3D()
checkpoint =  torch.load(checkpoint_name)
model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
model.eval() # ensure we are in evaluation (not training mode)

torch.manual_seed(0)
# latent dim may depend on the model
latent_dim = 64 # hardcoded for now
hdf_keyname = 'data'



for i in range(num_samples):
    # Save off the generated volume
    base_vol_filename = 'gen_' + str(i)
    file_name_vol = base_vol_filename
    a_path = os.path.join(file_path_generated_volume, file_name_vol)

#  This could be done with a batch instead
    batch_size_loc = 1
    rand_input = torch.randn(batch_size_loc * latent_dim ).view(batch_size_loc, latent_dim, 1, 1, 1)
    output = model.generator(rand_input).detach()
    output = torch.squeeze(output)
    print(str(torch.min(output))+str(torch.max(output)))
    bin_output = output > 0.5  # make binary {0,1} by rounding
    with h5py.File(a_path+'.hdf5' , 'w') as f:
        f.create_dataset(hdf_keyname, data=bin_output, dtype="i8", compression="gzip")

#    img = output.numpy()
    if i==0: # load first image and make slices
        img = None
        filepath = a_path+'.hdf5'
        with h5py.File(filepath, "r") as f:
            img = f[hdf_keyname][()]
        massaged_output = img
        print( '{:0.2f}±{:0.2f}, [{:0.2f},{:0.2f}]'.format(np.mean(img),np.std(img),np.min(img),np.max(img)))
        for j in range(len(massaged_output)):
            file = a_path+'_binaryslice_'+str(j)+'.png'
            imageio.imwrite(file, np.around(massaged_output[j]))

        file = a_path + '_binaryave' + '.png'
        imageio.imwrite(file, np.mean(np.around(massaged_output),axis=0))



total_time = time.time() - ts_start

print("Generation of volumes done.")
print("Wall time expended: " + str(total_time) + " seconds.")
print("Generated volumes are in " + file_path_generated_volume)
print("Now run analysis with GooseEYE installed.")

### END: Generate an output for a saved model #######################################################################################

