import matplotlib.pyplot as plt
import os
import numpy as np
#from sklearn.ensemble._hist_gradient_boosting import loss
#from lpdgen.models import DCGAN3D
import GooseEYE 
import time
#import torch.nn as nn
#import torch.nn.parallel
#import pytorch_lightning as lightning
import glob
import math

PNGs_wanted = False

###  Generate several outputs for a saved model, then run analysis against it #######################################################################################

ts_start = time.time()
print("Model analysis starting - mathematical analysis.")

#alpha_const = 0.7
#alpha_const = 0.02
alpha_const = 0.05

timestamp1 = str(round(ts_start))        # Makes it harder to accidentally overwrite things

print("Timestamp: " + timestamp1)

# Where to read the generated volumes (the artificial materials)
file_path_generated_volume = '../analysis/gen_volumes/' 

##########################################################################################

input_files = glob.glob(os.path.join(file_path_generated_volume, '*.npy'))

# Where to save the analysis
file_path_analysis = '../analysis/results/' 
file_name_analysis = timestamp1 + '.txt'
file_results = open(os.path.join(file_path_analysis, file_name_analysis), '+w')

file_results.write('Mathematical analysis run')
file_results.write('\n')
file_results.write('Timestamp: ' + timestamp1)
file_results.write('\n')
file_results.write('==================================')
file_results.write('\n')

# Lineal path function setup
lp_sum = 0

file_results.write("Lineal path function - mean of max ===========================================")
file_results.write('\n')

for f in input_files:
    incoming_file = np.load(f)
    
    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(incoming_file)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate lineal path function for this generated volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    lp_sum = 0
    for i in range(volume_size):
        lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
        lp_sum += max(lpf)

    # Write out the lineal path function result for this generated volume.    
    lp_mean = lp_sum/volume_size
    file_results.write(f + "',  value: " + str(1 - lp_mean[0]))
    file_results.write('\n')

    print(f + ", " + str(1 - lp_mean[0]))
    ######################## Done calculating the lineal path function.

file_results.write("")
file_results.write('\n')
file_results.write("2-point probability function - mean of max ===========================================")
file_results.write('\n')

for f in input_files:
    incoming_file = np.load(f)
    
    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(incoming_file)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate 2-point probability function for this generated volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    sum = 0
    for i in range(volume_size):  
        fn_goose = GooseEYE.S2((line_length, 1), massaged_output[i], massaged_output[i]) #31 is the distance over two points(white and black), (1, 31) gives us (x, y) different axis
        sum += max(fn_goose)
        
    mean = sum/volume_size
    #find the mean of max value: the mean (1 - porosity) in the third dimension
    
    file_results.write(f + "',  value: " + str(1 - mean[0]))
    file_results.write('\n')
    
    print(f + ", " + str(1 - mean[0]))
    ######################## Done calculating the 2-point probability function.

# Lineal path function setup - MEAN
lp_sum = 0

file_results.write("Lineal path function - mean of ALL ===========================================")
file_results.write('\n')

for f in input_files:
    incoming_file = np.load(f)
    len_fn_output = 0
    
    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(incoming_file)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate mean of the lineal path function for this generated volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    lp_sum = 0
    for i in range(volume_size):
        lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
        lp_sum += np.sum(lpf)
        len_fn_output = len(lpf)

    # Write out the lineal path function result for this generated volume.    
    lp_mean = (lp_sum/len_fn_output)/volume_size
    file_results.write(f + "',  value: " + str(lp_mean))
    file_results.write('\n')

    print(f + ", " + str(lp_mean))
    ######################## Done calculating the lineal path function.

file_results.write("")
file_results.write('\n')
file_results.write("2-point probability function - mean of ALL ===========================================")
file_results.write('\n')

for f in input_files:
    incoming_file = np.load(f)
    
    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(incoming_file)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate 2-point probability function for this generated volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    sum_fn = 0
    for i in range(volume_size):  
        fn_goose = GooseEYE.S2((line_length, 1), massaged_output[i], massaged_output[i]) #31 is the distance over two points(white and black), (1, 31) gives us (x, y) different axis
        sum_fn += np.sum(fn_goose)
        len_fn_output = len(fn_goose)
        
    mean = (sum_fn/len_fn_output)/volume_size    
    file_results.write(f + "',  value: " + str(mean))
    file_results.write('\n')
    
    print(f + ", " + str(mean))
    ######################## Done calculating the 2-point probability function.

if PNGs_wanted:
    # Make PNG images from the volumes
    print("Printing PNGs")
    for f in input_files:
        incoming_file = np.load(f)
        
        print("processing file " + f)
        
        # Force to exactly zero and exactly one so we have a binary volume (material and void).
        massaged_output = np.squeeze(incoming_file)
        massaged_output = np.around(massaged_output)
        
        # Save as an image so we can look at it if we need to.
        z,x,y = massaged_output.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c='grey', alpha=alpha_const)
        file_name_img = os.path.basename(f) + '.png'
        plt.savefig(os.path.join(file_path_generated_volume, file_name_img), dpi=300)
        plt.close()
    

total_time = time.time() - ts_start

file_results.write('\n')
file_results.write('==================================')
file_results.write('\n')
file_results.write("Wall time expended: " + str(total_time) + " seconds.")
file_results.write('\n')
file_results.close()

print("Model analysis done.")
print("Wall time expended: " + str(total_time) + " seconds.")

### END: Generate an output for a saved model #######################################################################################

