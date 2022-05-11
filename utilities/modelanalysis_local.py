import matplotlib.pyplot as plt
import os
import numpy as np
import GooseEYE 
import time
import glob

###  Generate several outputs for a saved model, then run analysis against it #######################################################################################

ts_start = time.time()
print("Model analysis starting - mathematical analysis.")

#alpha_const = 0.7
alpha_const = 0.05

timestamp1 = str(round(ts_start))        # Makes it harder to accidentally overwrite things

print("Timestamp: " + timestamp1)

# Where to read the generated volumes (the artificial materials)
file_path_generated_volume = '../../analysis/gen_volumes/' 

# Where to save the analysis
file_path_analysis = '../../analysis/results/' 
file_name_analysis = run_num + '_numerical_analysis_' + epoch_num + '_' + timestamp1 + '.txt'
file_results = open(os.path.join(file_path_analysis, file_name_analysis), 'w')

##########################################################################################

file_results.write('Mathematical analysis run')
file_results.write('Timestamp: ' + timestamp1)
file_results.write('==================================')

input_files = glob.glob(os.path.join(file_path_generated_volume, '*.npy'))

# Lineal path function setup
lp_sum = 0

for f in input_files:
    incoming_file = np.load(f)

    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(incoming_file)
    massaged_output = np.around(massaged_output)
    
    # Save as an image so we can look at it if we need to.
    z,x,y = massaged_output.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='grey', alpha=alpha_const)
    file_name_img = base_vol_filename + '.png'
    plt.savefig(os.path.join(file_path_generated_volume, file_name_img))
    
    ######################## Calculate lineal path function for this generated volume.
    volume_size = 64        # Volume is 64 voxels across.
    for i in range(volume_size):
        lpf = GooseEYE.L((50, 1), massaged_output[i])  # 50 is the length of the line, (1, 50) gives us (x, y) different axis
        lp_sum += max(lpf)

    # Write out the lineal path function result for this generated volume.    
    lp_mean = lp_sum/volume_size
    file_results.write("Lineal path function, volume: '" + f + "',  value: " + str(lp_mean))
    ######################## Done calculating the lineal path function.

total_time = time.time() - ts_start

file_results.write()
file_results.write("Wall time expended: " + str(total_time) + " seconds.")
file_results.close()

print("Model analysis done.")
print("Wall time expended: " + str(total_time) + " seconds.")

### END: Generate an output for a saved model #######################################################################################

