import os
import numpy as np
from ..lpdgen.models import DCGAN3D
import time
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning

###  Generate several outputs for a saved model, then run analysis against it #######################################################################################

ts_start = time.time()
print("Generation of volumes on HPC starting.")

# Set these.
run_num = '889570'
epoch_num = '44'
num_samples = 644;       # Number of samples we make from the model.

timestamp1 = str(round(ts_start))        # Makes it harder to accidentally overwrite things

# Where to load model from
file_path_models = '/home/2578/gore1/867-team-gore/analysis/models/'
file_name_model = run_num + '_model_and_opt_save_' + epoch_num + '.torchsave'

# Where to save the generated volumes (the artificial materials)
file_path_generated_volume = '/home/2578/gore1/867-team-gore/analysis/gen_volumes/' 

# Analysis now done on local PC
## Where to save the analysis
#file_path_analysis = '/home/2578/gore1/867-team-gore/analysis/results/' 
#file_name_analysis = run_num + '_numerical_analysis_' + epoch_num + '_' + timestamp1 + '.txt'
#file_results = open(os.path.join(file_path_analysis, file_name_analysis), 'w')

##########################################################################################

print('Run Number: ' + run_num)
print('Epoch: ' + epoch_num)
print('Timestamp: ' + timestamp1)
print('==================================')

model = DCGAN3D()
checkpoint =  torch.load(os.path.join(file_path_models, file_name_model))
model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
model.eval() 

# Can't do this on Darwin HPC - I can't load GooseEYE there.  Have to do it on local PC.
## Lineal path function setup
#lp_sum = 0

for i in range(num_samples):
    batch_size_loc = 1
    rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
    rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
    output = model.generator(rand_input)
    output = output.detach().numpy()
        
    # Save off the generated volume
    base_vol_filename = run_num + '_generatedvolume_' + epoch_num + '_' + timestamp1 +  '_' + str(i)
    file_name_vol = base_vol_filename + '.npy'
    np.save(os.path.join(file_path_generated_volume, file_name_vol), output)

    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(output)
    massaged_output = np.around(massaged_output)
    

total_time = time.time() - ts_start

print("Generation of volumes done.")
print("Wall time expended: " + str(total_time) + " seconds.")
print("Generated volumes are in " + file_path_generated_volume)
print("Now run utilities/modelanalysis_local on a PC with GooseEYE installed.")

### END: Generate an output for a saved model #######################################################################################

