import matplotlib.pyplot as plt
import os
import numpy as np
import time
import glob
from hdf5storage import loadmat
from numpy.ma.bench import xs
import math
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning
from models import DCGAN3D
from scipy.stats import gaussian_kde
import GooseEYE 
from scipy.stats import ks_2samp        # Kolmogorov-Smirnov test for two sets of data; determines how likely two sets of data are to be from the same distribution
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



### Make figures needed for project report and presentation. #######################################################################################
###
### Run this on the local machine, not on Darwin.


# Change these, if needed ###########################################
run_number_synth = '1651392030'     # A, 895488, epoch 245
#run_number_synth = '1651391562'     # B, 895489, epoch 207
run_number_real = '1651134843'
file_path_in = '../analysis/figures_in/'
file_path_out = '../analysis/figures_out/'
file_in_synth = os.path.join(file_path_in, run_number_synth + '.txt')
file_in_real = os.path.join(file_path_in, run_number_real + '_analysis_of_held_out.txt')
#alpha_const = 0.7
alpha_const = 0.05


ts_start = math.trunc(time.time())
print("Start - making figures for report and presentation.")

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def get_lines_for_graph(wanted_format, file_wanted):
    # This is ugly but it's only going to be used a couple of times.
    linpath_porosity_read = False
    twopt_porosity_read = False
    linpath_mean_read = False
    twopt_mean_read = False
    
    lines = []
    
    f = open(file_wanted, 'r')
    for linea in f:
        if wanted_format == 2 and (linea.strip() == "Lineal path function - mean of max ===========================================" or linea.strip() == "Lineal path function - porosity (mean of max) ==========================================="):
            linpath_porosity_read = True
            twopt_porosity_read = False
            linpath_mean_read = False
            twopt_mean_read = False
            continue
        if wanted_format == 3 and (linea.strip() == "2-point probability function - mean of max ===========================================" or linea.strip() == "2-point probability function - porosity (mean of max) ==========================================="):
            linpath_porosity_read = False
            twopt_porosity_read = True
            linpath_mean_read = False
            twopt_mean_read = False
            continue
        if wanted_format == 4 and (linea.strip() == "Lineal path function - mean of ALL ===========================================" or linea.strip() == "Lineal path function - mean of all ==========================================="):
            linpath_porosity_read = False
            twopt_porosity_read = False
            linpath_mean_read = True
            twopt_mean_read = False
            continue
        if wanted_format == 5 and (linea.strip() == "2-point probability function - mean of ALL ===========================================" or linea.strip() == "2-point probability function - mean of all ==========================================="):
            linpath_porosity_read = False
            twopt_porosity_read = False
            linpath_mean_read = False
            twopt_mean_read = True
            continue        
        if wanted_format == 2 and linpath_porosity_read:
            if linea.strip() == "" or linea.strip()[-1] == "=":
                linpath_porosity_read = False
                continue
            else:
                lines.append(linea)
                continue
        if wanted_format == 3 and twopt_porosity_read:
            if linea.strip() == "" or linea.strip()[-1] == "=":
                twopt_porosity_read = False
                continue
            else:
                lines.append(linea)
                continue            
        if wanted_format == 4 and linpath_mean_read:
            if linea.strip() == "" or linea.strip()[-1] == "=":
                linpath_mean_read = False
                continue
            else:
                lines.append(linea)
                continue                   
        if wanted_format == 5 and twopt_mean_read:
            if linea.strip() == "" or linea.strip()[-1] == "=":
                twopt_mean_read = False
                continue
            else:
                lines.append(linea)
                continue             
    
    f.close()
    
    return lines


def make_graph_from_file(wanted_format, file_synthetic, file_real, file_out_hist, file_out_line, graph_title, graph_x_label, graph_y_label):
    lines = get_lines_for_graph(wanted_format, file_synthetic)
    
    value_list = []
    bin_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    buckets_synthetic = [0]*len(bin_list)
    
    buckets_synthetic_line = [0]*(len(bin_list)-1)
    
    for l in lines:
        # Get the actual value out of the line of text.
        decimalpt_loc = l.rfind('.')
        fraction_val = float(l[(decimalpt_loc-1):])   # Get just the number. It (should) be 0 <= n <= 1, but if it's 0 <= n < 10, we're okay.
        value_list.append(fraction_val)
    
        # Put the value into a bucket.
        for i in range(len(bin_list) - 1):
            if fraction_val >= bin_list[i] and fraction_val < bin_list[i+1]:
                buckets_synthetic_line[i] = buckets_synthetic_line[i] + 1
            
            if fraction_val == 1:
                    buckets_synthetic_line[-1] = buckets_synthetic_line[-1] + 1
                
            if fraction_val > 1:
                print("PROBLEM!  We have a value greater than 1. " + str(fraction_val))   # Should never happen, but is here to be careful.
    
    lines = get_lines_for_graph(wanted_format, file_real)
    
    value_list_real = []
    buckets_real = [0]*len(bin_list)
    
    buckets_real_line = [0]*(len(bin_list)-1)
    buckets_x_line = [0]*(len(bin_list)-1)
    
    # Used to make the graph itself (center of the bucket's range)
    for i in range(len(buckets_x_line)):
        buckets_x_line[i] = (bin_list[i] + bin_list[i+1]) / 2
    
    for l in lines:
        # Get the actual value out of the line of text.
        decimalpt_loc = l.rfind('.')
        fraction_val = float(l[(decimalpt_loc-1):])   # Get just the number. It (should) be 0 <= n <= 1, but if it's 0 <= n < 10, we're okay.
        value_list_real.append(fraction_val)

        # Put the value into a bucket.
        for i in range(len(bin_list) - 1):
            if fraction_val >= bin_list[i] and fraction_val < bin_list[i+1]:
                buckets_real_line[i] = buckets_real_line[i] + 1
            
            if fraction_val == 1:
                    buckets_real_line[-1] = buckets_real_line[-1] + 1
                
            if fraction_val > 1:
                print("PROBLEM!  We have a value greater than 1. " + str(fraction_val))
    
    print("Generating histogram...")
    
    plt.hist([value_list_real, value_list], bin_list, label=['Test Data Samples', 'Model-Generated Samples'], color=['black', 'red'])
    plt.ylabel(graph_y_label)
    plt.xlabel(graph_x_label)
    plt.title(graph_title)
    
    if wanted_format == 2 or wanted_format == 3:
        legend_pos = 'upper right'
    else:
        legend_pos = 'upper left'
        
    plt.legend(loc=legend_pos)
    plt.savefig(file_out_hist, dpi=300)
    print("Saved to " + file_out_hist)
    #plt.show()  
    plt.clf()
    
    print("Generating line graph...")
    plt.plot(buckets_x_line, buckets_real_line, label="Test Data Samples", marker='o', color='black')
    plt.plot(buckets_x_line, buckets_synthetic_line, label="Model-Generated Samples", marker='x', color='red')
    plt.ylabel(graph_y_label)
    plt.xlabel(graph_x_label)
    plt.title(graph_title)
    plt.legend(loc=legend_pos)
    plt.savefig(file_out_line, dpi=300)
    print("Saved to " + file_out_line)
    #plt.show()
    plt.clf()


def make_graph_from_lists(model_vals, test_vals, file_out_hist, file_out_line, graph_title, graph_x_label, graph_y_label, legend_pos):
    bin_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    buckets_synthetic = [0]*len(bin_list)
    buckets_synthetic_line = [0]*(len(bin_list)-1)
    
    value_list = model_vals
    
    for fraction_val in model_vals:
        # Put the value into a bucket.
        for i in range(len(bin_list) - 1):
            if fraction_val >= bin_list[i] and fraction_val < bin_list[i+1]:
                buckets_synthetic_line[i] = buckets_synthetic_line[i] + 1
            
            if fraction_val == 1:
                    buckets_synthetic_line[-1] = buckets_synthetic_line[-1] + 1
                
            if fraction_val > 1:
                print("PROBLEM!  We have a value greater than 1. " + str(fraction_val))   # Should never happen, but is here to be careful.

    value_list_real = test_vals
    buckets_real = [0]*len(bin_list)
    
    buckets_real_line = [0]*(len(bin_list)-1)
    buckets_x_line = [0]*(len(bin_list)-1)
    
    # Used to make the graph itself (center of the bucket's range)
    for i in range(len(buckets_x_line)):
        buckets_x_line[i] = (bin_list[i] + bin_list[i+1]) / 2
    
    for fraction_val in test_vals:
        # Put the value into a bucket.
        for i in range(len(bin_list) - 1):
            if fraction_val >= bin_list[i] and fraction_val < bin_list[i+1]:
                buckets_real_line[i] = buckets_real_line[i] + 1
            
            if fraction_val == 1:
                    buckets_real_line[-1] = buckets_real_line[-1] + 1
                
            if fraction_val > 1:
                print("PROBLEM!  We have a value greater than 1. " + str(fraction_val))
    
    print("Generating histogram...")
    plt.hist([value_list_real, value_list], bin_list, label=['Test Data Samples', 'Model-Generated Samples'], color=['black', 'red'])
    plt.ylabel(graph_y_label)
    plt.xlabel(graph_x_label)
    plt.title(graph_title)
    plt.legend(loc=legend_pos)
    plt.savefig(file_out_hist, dpi=600)
    print("Saved to " + file_out_hist)
    plt.clf()
    
    print("Generating line graph...")
    plt.plot(buckets_x_line, buckets_real_line, label="Test Data Samples", marker='o', color='black')
    plt.plot(buckets_x_line, buckets_synthetic_line, label="Model-Generated Samples", marker='x', color='red')
    plt.ylabel(graph_y_label)
    plt.xlabel(graph_x_label)
    plt.title(graph_title)
    plt.legend(loc=legend_pos)
    plt.savefig(file_out_line, dpi=600)
    print("Saved to " + file_out_line)
    plt.clf()


def get_lists_of_saved_models(model_loc, outfile_loc, start_run):
    # Returns a triplet of lists: epoch_numbers, run_numbers, run_epoch_numbers
    #    
    #  epoch_numbers:      Incremental epoch numbers, for the entire training 
    #  run_numbers:        The run which happened during this training epoch
    #  run_epoch_numbers:  The number recorded BY THE RUN for the epoch (not the same as the training epoch)
    
    done_flag= False
    
    run_list = []
    epoch_list = []
    model_file_list = []
    
    epoch_numbers = []
    run_numbers = []
    run_epoch_numbers = []
    
    training_epoch = 0
    prev_run = start_run   
    
    while not done_flag:
        current_run = prev_run
        file_in = os.path.join(outfile_loc, 'slurm-' + str(current_run) + '.out')
                
        epoch = -1
    
        # Get the previous run.
        f = open(file_in, 'r', encoding="utf8")

        cleaned_lines = []

        # Get lines of the file with non-ASCII chars removed, put them in a list.
        for l in f:
            l = l.strip()
            
            # Get rid of non-ascii characters
            encoded_str = l.encode("ascii", "ignore")
            l = encoded_str.decode()
            
            cleaned_lines.append(l)
    
        # Get the previous run's run number
        validating = False
        for linea in cleaned_lines:
            if len(linea) >= 18:
                if linea[0:17] == 'Location of model':
                    # Okay, we found the location of the previous run's file.  Get the previous run number.
                    slash_loc = linea.rfind('/')
                    prev_run_filename = (linea[(slash_loc+1):])
                    prev_run = int(prev_run_filename[:6])
                                        
                    run_list.append(prev_run)
                   
        if prev_run == current_run:
            done_flag = True        # We didn't find a previous run in this outfile.  Done.
            #print('Previous run not found in run ' + str(current_run))    # Done.
        
    file_in = os.path.join(model_loc, 'slurm-' + str(current_run) + '.out')
    
    run_list.sort()
    
    all_models_found = []
    
    for r in run_list:
        models_found = glob.glob(os.path.join(model_loc + str(r) + "*.torchsave"))
        
        ep = []
        
        # Sort by epoch
        for m in models_found:
            dot_found = m.rfind(".")
            uline_found = m.rfind("_")
            ep.append(int(m[uline_found+1:dot_found]))
        
        z = zip(ep, models_found)
        z2 = list(z)
        z2.sort()
        
        if len(z2) > 0:
            ep, models_found = zip(*z2)
        else:
            models_found = []
        
        for m in models_found:
            all_models_found.append(m)
    
    universal_epoch = 0
    prev_univ_epoch = -1
    prev_local_epoch = -1
    
    for m in all_models_found:
        dot_found = m.rfind(".")
        uline_found = m.rfind("_")
        local_epoch = int(m[uline_found+1:dot_found])
        
        bslash_found = m.rfind("\\")
        local_run = int(m[bslash_found+1:bslash_found+7])
        
        if prev_local_epoch > local_epoch:
            # New set of local epoch counts found
            if local_epoch == 0:
                universal_epoch += 1
            else:
                universal_epoch += local_epoch
        else:
            # Still in an existing run of local epoch counts
            epoch_diff = local_epoch - prev_local_epoch
            universal_epoch += epoch_diff
         
        prev_local_epoch = local_epoch
        
        epoch_numbers.append(universal_epoch)
        run_epoch_numbers.append(local_epoch)
        run_numbers.append(local_run)

    return [run_numbers, epoch_numbers, run_epoch_numbers]
    

def get_ground_truth_subvolumes(file_path_input):
    """
    We'll make as many subvolumes as we can from the main volume of every sample. Our
    requirement is that a subvolume needs at least 2 void (pore) voxels.
    
    file_path_input: path to the data; files will be in MAT format.
    """
    
    chunksize = 64           # Size of the block we want to extract from the full-sized voxel "image"
    sizeoforiginal = 256     # Size of the full-sized image
    
    filelist = os.listdir(file_path_input)
    
    # Load data into memory
    ready_data = []
    
    requiredpore = 2        # how many pore voxels do we require?
    
    for index in range(len(filelist)):
        # Try to make subvolumes for every file's data
        itm = loadmat(os.path.join(file_path_input, filelist[index]))['bin']
        
        for newleftA in range(itm.shape[0] // chunksize):
            for newfrontA in range(itm.shape[0] // chunksize):
                for newtopA in range(itm.shape[0] // chunksize):
                    newleft = newleftA * chunksize
                    newfront = newfrontA * chunksize
                    newtop = newtopA * chunksize
                    subset_of_item = itm[newleft:newleft+chunksize, newtop:newtop+chunksize, newfront:newfront+chunksize]
                    
                    numporevoxels = ((subset_of_item.shape[0])**3) - np.sum(subset_of_item)
                    
                    if (numporevoxels > requiredpore):
                        # Store the file name in location for later use
                        ready_data.append(subset_of_item)
    
    return ready_data



def porosity_via_lineal_path(subvolume):
    # Lineal path to give porosity (1 - [mean of max of lineal path])
    lp_sum = 0
    
    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(subvolume)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate mean of max of lineal path function for this data volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    lp_sum = 0
    for i in range(volume_size):
        lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
        lp_sum += max(lpf)

    # return the porosity as the 1 minus (mean of the max of the lineal path function result) for this generated volume.    
    return 1 - (lp_sum/volume_size)


def lineal_path_mean_all(subvolume):
    # mean of the lineal path function 

    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(subvolume)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate lineal path function for this data volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    lp_sum = 0
    fn_len = 0
    for i in range(volume_size):
        lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
        lp_sum += np.sum(lpf)
        fn_len = len(lpf)

    # Write out the lineal path function result for this volume.    
    mean = (lp_sum/fn_len)/volume_size
    return mean


def two_pt_prob_mean_all(subvolume):
    # Force to exactly zero and exactly one so we have a binary volume (material and void).
    massaged_output = np.squeeze(subvolume)
    massaged_output = np.around(massaged_output)
    
    ######################## Calculate 2-point probability function for this volume.
    volume_size = 64        # Volume is 64 voxels across.
    line_length = math.trunc(volume_size / 2) - 1
    
    sum = 0
    fn_len = 0            
    for i in range(volume_size):  
        fn_goose = GooseEYE.S2((line_length, 1), massaged_output[i], massaged_output[i]) #31 is the distance over two points(white and black), (1, 31) gives us (x, y) different axis
        sum += np.sum(fn_goose)
        fn_len = len(fn_goose)
        
    mean = (sum/fn_len)/volume_size
    return mean
    #find the mean all values
    ######################## Done calculating the 2-point probability function.   


#################################################################
#################################################################
#################################################################
#################################################################

timestamp1 = str(round(ts_start))        # Makes it harder to accidentally overwrite things

print("Timestamp: " + timestamp1)


print("1 - Print 256-volume with planes (ugly; do not use)")
print("2 - Create graphs of analysis data - porosity as calculated by lineal path function")
print("3 - Create graphs of analysis data - porosity as calculated by 2-point function")
print("4 - Create graphs of analysis data - mean of lineal path function")
print("5 - Create graphs of analysis data - mean of 2-point function")
print("")
print("9 - All of options 2 to 5")
print("10 - Compare lineal path and 2-point function values")
print("11 - Graph of loss values")
print("12 - Kolmogorov-Smirnov analysis")
print("13 - ")
print("14 - Graph of K-S analysis")
print("15 - Graph of K-S analysis with error bars")
print("16 - Image as voxels, and print out its noise vector")
print("17 - Input a noise vector, output it as voxels")
print("18 - Noise vector over time")
print("19 - One model with sliding noise vector")
print("20 - Print 256-volume divided into subvolumes, using voxels")
print("21 - Print 256-volume NOT divided, using voxels")
print("22 - Print 64-subvolume as voxels")
print("23 - Graph of K-S p-values with error bars")
print("24 - Model selection - best p-values")
print("25 - Model selection - best K/S distances")
print("26 - Getting some k/s info for epoch 38")
print("27 - Epoch 38: Generating volumes and outputting as voxels")
print("28 - Outputting chunks of test data (ground truth)")
print("30 - Make analysis data, starting with a saved model and the held-out test data")
print("31 - Make graphs from the data generated in option 30.")
print('32 - Feature explanatory images')

req_option = int(input("Enter desired option: "))

if req_option == 0:
    print("That was not one of the options.")

    ####################################################

elif req_option == 1:
    file_path_out_loc = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/256_box_with_slices/'
    
    file_in = os.path.join(file_path_in, '10_03_256.mat')
    file_out = os.path.join(file_path_out_loc, timestamp1 + '_256_with_division_planes.png')
    alpha_const = 0.05

    nowww = time.time()
    
    print("Printing 256-volume with planes.  Input should be at " + file_in)
    print("Generating image...")
    
    incoming_data = loadmat(file_in)['bin']
    z,x,y = incoming_data.nonzero()

    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')
    ax.voxels(incoming_data, facecolors='oldlace', edgecolor='k', linewidth=0.2)
    #plt.show()
    
    print("Wallclock seconds after making voxel image: " + str(time.time() - nowww))
    
    for mult_i in range(1,4):
        xs = np.linspace(start = -5, stop = 260, num = 200)
        ys = np.linspace(start = -5, stop = 260, num = 200)
        xs, ys = np.meshgrid(xs, ys)
        zs = xs*0 + ys*0 + 64*mult_i
        ax.plot_surface(xs, ys, -zs, color='red')

    for mult_i in range(1,4):
        xs = np.linspace(start = -5, stop = 260, num = 200)
        ys = np.linspace(start = -5, stop = 260, num = 200)
        xs, ys = np.meshgrid(xs, ys)
        zs = xs*0 + mult_i*64
        ax.plot_surface(zs, ys, -xs, color='r')

    for mult_i in range(1,4):
        xs = np.linspace(start = -5, stop = 260, num = 200)
        ys = np.linspace(start = -5, stop = 260, num = 200)
        xs, ys = np.meshgrid(xs, ys)
        zs = xs*0 + mult_i*64
        ax.plot_surface(xs, zs, -ys, color='r')

    plt.savefig(file_out, dpi=300)
    plt.close()
    print("Figure saved to " + file_out)
    
    ###########################################################################################

elif req_option == 2:
    print("Generating graph of analysis data.  Inputs should be at " + file_in_synth + " and " + file_in_real)
    
    file_hist = os.path.join(file_path_out, str(ts_start) + "_histogram_porosity_mean_lineal_path_function_ALL.png")
    file_line = os.path.join(file_path_out, str(ts_start) + "_line_porosity_mean_lineal_path_function_ALL.png")
    
    make_graph_from_file(2, file_in_synth, file_in_real, file_hist, file_line, "Porosity as Calculated Using Lineal Path Function", "Porosity", "Sample Count")
   
    print("Figures of analysis data done.")

    ###########################################################################################
    
elif req_option == 3:
    print("Generating graph of analysis data.  Inputs should be at " + file_in_synth + " and " + file_in_real)
    
    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_porosity_mean_2_pt_prob_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_porosity_mean_2_pt_prob_function_ALL.png")
    
    make_graph_from_file(3, file_in_synth, file_in_real, file_hist, file_line, "Porosity as Calculated Using Two-Point Probability Function", "Porosity", "Sample Count")
   
    print("Figures of analysis data done.")
    
    ###########################################################################################

elif req_option == 4:
    print("Generating graph of analysis data.  Inputs should be at " + file_in_synth + " and " + file_in_real)
    
    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_mean_lineal_path_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_mean_lineal_path_function_ALL.png")
    
    make_graph_from_file(4, file_in_synth, file_in_real, file_hist, file_line, "Count of Samples by Mean of Lineal Path Function", "Mean of Lineal Path", "Sample Count")
   
    print("Figures of analysis data done.")
    
    ###########################################################################################

elif req_option == 5:
    print("Generating graph of analysis data.  Inputs should be at " + file_in_synth + " and " + file_in_real)
    
    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_mean_2_pt_probability_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_mean_2_pt_probability_function_ALL.png")
    
    make_graph_from_file(5, file_in_synth, file_in_real, file_hist, file_line, "Count of Samples by Mean of Two-Point Probability Function", "Mean of Two-Point Probability", "Sample Count")
   
    print("Figures of analysis data done.")
    
    ###########################################################################################

elif req_option == 9:
    print("Generating graph of analysis data.  Inputs should be at " + file_in_synth + " and " + file_in_real)

    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_porosity_mean_lineal_path_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_porosity_mean_lineal_path_function_ALL.png")
    
    make_graph_from_file(2, file_in_synth, file_in_real, file_hist, file_line, "Porosity as Calculated Using Lineal Path Function", "Porosity", "Sample Count")

    ##

    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_porosity_mean_2_pt_prob_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_porosity_mean_2_pt_prob_function_ALL.png")
    
    make_graph_from_file(3, file_in_synth, file_in_real, file_hist, file_line, "Porosity as Calculated Using Two-Point Probability Function", "Porosity", "Sample Count")

    ##

    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_mean_lineal_path_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_mean_lineal_path_function_ALL.png")
    
    make_graph_from_file(4, file_in_synth, file_in_real, file_hist, file_line, "Count of Samples by Mean of Lineal Path Function", "Mean of Lineal Path", "Sample Count")

    ##

    file_hist = os.path.join(file_path_out, str(run_number_synth) + "_histogram_mean_2_pt_probability_function_ALL.png")
    file_line = os.path.join(file_path_out, str(run_number_synth) + "_line_mean_2_pt_probability_function_ALL.png")
    
    make_graph_from_file(5, file_in_synth, file_in_real, file_hist, file_line, "Count of Samples by Mean of Two-Point Probability Function", "Mean of Two-Point Probability", "Sample Count")
       
    print("Figures of analysis data done.")
    
    ###########################################################################################
        
elif req_option == 10:
    print("*** Checking to see if 2-pt and lineal path are the same ***")
    
    A_file_in = os.path.join(file_path_in, 'A_1651115764.txt')
    B_file_in = os.path.join(file_path_in, 'B_1651115764.txt')
        
    A_f = open(A_file_in, 'r')
    A_lines = A_f.readlines()
    
    B_f = open(B_file_in, 'r')
    B_lines = B_f.readlines()
    
    for i in range(len(A_lines)):
        if A_lines[i] != B_lines[i]:
            print(A_lines[i])
            print(B_lines[i])
            print("========")
    
    A_f.close()
    B_f.close()
    
    ###########################################################################################
        
elif req_option == 11:
    print("*** Making loss graph ***")
    
    file_path_in = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/out_files/'
    file_path_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/loss/'
    
    plot_aspect = 3.83 / 6.94
    plot_width = 10
    plot_height = plot_width * plot_aspect
    
    timestampA = str(int(time.time()))
    
    ## Get the nnnnnn part of slurm-nnnnnn.out from the user. 
    #last_run = input("What is the number of the last run? ")
    #prev_run = int(last_run)
    #
    #log_scale_str = input("Do you want the plot to be in log scale? (T/F) ")
    #log_scale = (log_scale_str.upper() == "T")
    
    prev_run = 999999   # Main run (the one we did all the work on)
    #prev_run = 896196   # Second run (the shorter one, used to show that GANs are really not gonna work)
    log_scale = True        # We know this is true
    
    file_out = os.path.join(file_path_out, str(prev_run) + "_" + str(timestampA) + '_losses' + '.png')
    
    losses_list = []
    epochs_list = []
    done_flag= False
    best_loss = 1000000   
    best_loss_epoch = 0
    best_loss_run= 0 
    
    biggest_loss_change = 0
    biggest_loss_change_epoch = 0
    biggest_loss_change_run = 0
    prev_loss = math.inf
    
    while not done_flag:
        current_run = prev_run
        prev_epoch = 0
        file_in = os.path.join(file_path_in, 'slurm-' + str(current_run) + '.out')
        epoch = -1
        
        run_losses = []
        run_epoch= []
    
        # Get the previous run.
        f = open(file_in, 'r', encoding="utf8")

        cleaned_lines = []

        # Get lines of the file with non-ASCII chars removed, put them in a list.
        for l in f:
            l = l.strip()
            
            # Get rid of non-ascii characters
            encoded_str = l.encode("ascii", "ignore")
            l = encoded_str.decode()
            
            cleaned_lines.append(l)
    
        # Get the previous run's run number and ending epoch
        validating = False
        for linea in cleaned_lines:
            if len(linea) >= 18:
                if linea[0:17] == 'Location of model':
                    # Okay, we found the location of the previous run's file.  Get the previous run number.
                    slash_loc = linea.rfind('/')
                    prev_run_filename = (linea[(slash_loc+1):])
                    prev_run = int(prev_run_filename[:6])
                    
                    # Get ending epoch
                    underl_loc = linea.rfind('_')
                    dot_loc = linea.rfind('.')
                    prev_epoch = int(linea[(underl_loc+1):dot_loc])
        
        # Get the end-of-epoch loss 
        for linea in cleaned_lines:
            
            # Find the end of validation.  Epoch loss is after this
            if linea[:5] == "Valid":
                percentage_loc = linea.find("%")
                if percentage_loc != -1:
                    percent_val = int(linea[(percentage_loc-3):percentage_loc])
                    if percent_val == 100:
                        validating = True
                        continue                
            
            if validating:
                # This is the line with the end of epoch loss
                percentage_loc = linea.find("%")
                if percentage_loc != -1:
                    percent_val = int(linea[(percentage_loc-3):percentage_loc])
                    if percent_val == 100:
                        first_colon = linea.find(":")
                        first_h = linea.find("h")
                        epoch = int(linea[(first_h+1):first_colon])
                        
                        if epoch != 0:
                            # Epoch zero is done during loading of the model by torch for some weird reason.
                            
                            new_ep = prev_epoch + epoch
                            run_epoch.insert(0, new_ep)
                        
                            # Okay, we now have the line with the epoch loss.  Get the loss.
                            last_comma = linea.rfind(",")
                            loss_text = linea.rfind("loss=")
                            strloss = linea[(loss_text+5):last_comma]
                            loss = abs(float(strloss))
                            if loss < best_loss:
                                best_loss = loss
                                best_loss_epoch = epoch
                                best_loss_run = current_run

                            if prev_loss != math.inf:
                                loss_change = abs(prev_loss - loss)
                                if loss_change >= biggest_loss_change:
                                    biggest_loss_change_epoch = epoch
                                    biggest_loss_change_run = current_run
                                    biggest_loss_change = loss_change

                            prev_loss = loss

                            run_losses.append(loss)
                
                validating = False
                     
        if prev_run == current_run:
            done_flag = True        # We didn't find a previous run in this outfile.  Done.
        
        f.close()

        for itm in run_epoch:
            epochs_list.insert(0, itm)
        
        for itm in run_losses:
            losses_list.insert(0, itm)
    
    corrected_epochs = []
    found_45 = False
    ep_offset = 0
    for itm in epochs_list:
        if itm == 45 and not found_45:
            found_45 = True
            ep_offset = 25
            corrected_epochs.append(itm)
        else:
            corrected_epochs.append(itm + ep_offset)
   
    print("")
    print("Best loss: " + str(best_loss))
    print("Epoch of best loss: " + str(best_loss_epoch))
    print("Run of best loss: " + str(best_loss_run))
    print("")    
    print("Biggest loss change: " + str(biggest_loss_change))
    print("Epoch of biggest loss change: " + str(biggest_loss_change_epoch))
    print("Run of biggest loss change: " + str(biggest_loss_change_run))
    print("")
    
    print("Generating line graph...")
    
    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)
    
    plt.plot(corrected_epochs, losses_list, label="Loss", marker='o', markersize=2, color='black')
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    
    #plt.plot(X, y_predict, color='red')         # Trendline
    
    if log_scale:
        plt.yscale('log')
    
    plt.title("Loss per Epoch")
    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    plt.show()
    #plt.clf()
    
    
elif req_option == 12:
    print("*** Making Kolmogorov-Smirnov analysis graph for many runs ***")
    
    file_path_in_models = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/model_backup/"
    file_path_in_logs = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/out_files/'
    file_path_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/'
    num_samples = 645
    
    #num_samples = 20        # Quick test

    ts_start = time.time()
    timestamp1 = str(round(ts_start))        # Makes it harder to accidentally overwrite things
    print("Timestamp: " + timestamp1)

    
    ##########################################################################################
    
    # Where to save the analysis
    file_name_analysis = timestamp1 + '_K-S_analysis.txt'
    file_results = open(os.path.join(file_path_out, file_name_analysis), '+w')
    
    file_results.write('Kolmogorov-Smirnov analysis')
    file_results.write('\n')
    file_results.write('Comparison of ground truth data and generated data using Kolmogorov-Smirnov to compare distributions.')
    file_results.write('\n')
    file_results.write('Timestamp: ' + timestamp1)
    file_results.write('\n')
    file_results.write('\n')


    # Get the ground truth data from DRP
    # You must have created the data file first!
    str_gt_porosity = get_lines_for_graph(2, file_in_real)
    str_gt_lineal_path_mean_of_all = get_lines_for_graph(4, file_in_real)
    str_gt_2_pt_prob_mean_of_all = get_lines_for_graph(5, file_in_real)
    
    gt_porosity = []
    gt_lineal_path_mean_of_all = []
    gt_2_pt_prob_mean_of_all = []
    
    for x in str_gt_porosity:
        decimalpt_loc = x.rfind('.')
        fraction_val = float(x[(decimalpt_loc-1):])   # Get just the number. It (should) be 0 <= n <= 1, but if it's 0 <= n < 10, we're okay.
        gt_porosity.append(fraction_val)

    for x in str_gt_lineal_path_mean_of_all:
        decimalpt_loc = x.rfind('.')
        fraction_val = float(x[(decimalpt_loc-1):])   # Get just the number. It (should) be 0 <= n <= 1, but if it's 0 <= n < 10, we're okay.
        gt_lineal_path_mean_of_all.append(fraction_val)

    for x in str_gt_2_pt_prob_mean_of_all:
        decimalpt_loc = x.rfind('.')
        fraction_val = float(x[(decimalpt_loc-1):])   # Get just the number. It (should) be 0 <= n <= 1, but if it's 0 <= n < 10, we're okay.
        gt_2_pt_prob_mean_of_all.append(fraction_val)
    # Done getting ground truth data.

    kolmogorov_smirnov = ks_2samp(gt_porosity, gt_porosity)

    print("gt_porosity vs itself, kolmogorov_smirnov.statistic, : " + str(kolmogorov_smirnov.statistic))
    print("gt_porosity vs itself, kolmogorov_smirnov.pvalue: "  + str(kolmogorov_smirnov.pvalue))
    print("===================================================================")
    file_results.write("gt_porosity vs itself, kolmogorov_smirnov.statistic, : " + str(kolmogorov_smirnov.statistic))
    file_results.write('\n')
    file_results.write("gt_porosity vs itself, kolmogorov_smirnov.pvalue: "  + str(kolmogorov_smirnov.pvalue))
    file_results.write('\n')
    file_results.write('=================================================================')
    file_results.write('\n')


    # Get the nnnnnn part of slurm-nnnnnn.out from the user. 
    last_run = input("What is the number of the last run? ")
    prev_run = int(last_run)

    model_info = get_lists_of_saved_models(file_path_in_models, file_path_in_logs, prev_run)

    model_run_numbers = model_info[0]    
    model_epoch_numbers = model_info[1]
    model_run_epoch_numbers = model_info[2]
    
    # Now process all the files we found.
    EPOCHS_lineal_path_mean_of_max_data = []
    EPOCHS_lineal_path_mean_of_all = []
    EPOCHS_two_pt_prob_mean_of_all = []
    
    # Store the final result for each epoch
    KS_lineal_path_mean_of_max_data = []
    KS_lineal_path_mean_of_all = []
    KS_two_pt_prob_mean_of_all = []

    ## Make a list of runs/epochs to test.
    # First, make a list of all runs/epochs
    mod_wanted = 10                  # Every 10th epoch as test
    run_list_full = []
    epoch_list_full = []
    run_list_chosen = []
    epoch_list_chosen = []
    univ_ep_chosen= []
    
    use_ep = -10000
    
    for i in range(len(model_run_epoch_numbers)):
        run_curr = model_run_numbers[i]
        ep = model_run_epoch_numbers[i]
        universal_ep = model_epoch_numbers[i]

        run_list_chosen.append(run_curr)
        epoch_list_chosen.append(ep)
        univ_ep_chosen.append(universal_ep)

    last_epoch = 0
    
    #for model_file_name in model_file_list:
    for i in range(len(epoch_list_chosen)):
        print(time.time())
        
        run_curr = run_list_chosen[i]
        epoch_curr = epoch_list_chosen[i]
        univ_epoch_curr = univ_ep_chosen[i]

        model_file_name = str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.torchsave'

        print(model_file_name)
        
        model = DCGAN3D()
        checkpoint =  torch.load(os.path.join(file_path_in_models, model_file_name))
        model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
        model.eval() 
        
        #### Lineal path function - mean of max (porosity)
        
        # Make samples from generator
        gen_samples = []        
        for i in range(num_samples):        
            batch_size_loc = 1
            rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
            rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
            output = model.generator(rand_input)
            output = output.detach().numpy()
                
            # Force to exactly zero and exactly one so we have a binary volume (material and void).
            massaged_output = np.squeeze(output)
            massaged_output = np.around(massaged_output)
            
            gen_samples.append(massaged_output)

        
        print(time.time())
        
        lineal_path_mean_of_max_data = []
        
        for massaged_output in gen_samples:        
            ######################## Calculate lineal path function for this generated volume.
            volume_size = 64        # Volume is 64 voxels across.
            line_length = math.trunc(volume_size / 2) - 1
            
            lp_sum = 0
            for i in range(volume_size):
                lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
                lp_sum += max(lpf)
        
            # Write out the lineal path function result for this generated volume.    
            lp_mean = lp_sum/volume_size
            
            lineal_path_mean_of_max_data.append(1 - lp_mean[0])
       
        kolmogorov_smirnov = ks_2samp(gt_porosity, lineal_path_mean_of_max_data)
        KS_lineal_path_mean_of_max_data.append(kolmogorov_smirnov)
        
        print("gt_porosity vs lineal_path_mean_of_max_data, kolmogorov_smirnov.statistic, universal epoch: "  + str(univ_epoch_curr) + ", run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.statistic))
        print("gt_porosity vs lineal_path_mean_of_max_data, kolmogorov_smirnov.pvalue, universal epoch: "  + str(univ_epoch_curr) + ", run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.pvalue))
        file_results.write("gt_porosity vs lineal_path_mean_of_max_data kolmogorov_smirnov.statistic, universal epoch: "  + str(univ_epoch_curr) + ", run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.statistic))
        file_results.write('\n')
        file_results.write("gt_porosity vs lineal_path_mean_of_max_data kolmogorov_smirnov.pvalue, universal epoch: "  + str(univ_epoch_curr) + ", run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.pvalue))
        file_results.write('\n')
        
        print(time.time())
        
        ###########################
    
        #### Lineal path function - mean of ALL
    
        #print("Lineal path function - mean of ALL ===========================================")
        
        lineal_path_mean_of_all = []
        
        for massaged_output in gen_samples:         
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
            
            lineal_path_mean_of_all.append(lp_mean)
            #print(str(lp_mean))
            ######################## Done calculating the lineal path function.  

        kolmogorov_smirnov = ks_2samp(gt_lineal_path_mean_of_all, lineal_path_mean_of_all)
        KS_lineal_path_mean_of_all.append(kolmogorov_smirnov)
        
        #print("Lineal path function - mean of max (porosity), run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " ===========================================")
        #print(kolmogorov_smirnov)

        print("gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.statistic, universal epoch: "  + str(univ_epoch_curr) + ",  run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.statistic))
        print("gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.pvalue, universal epoch: "  + str(univ_epoch_curr) + ",  run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.pvalue))
        file_results.write("gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.statistic, universal epoch: "  + str(univ_epoch_curr) + ",  run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.statistic))
        file_results.write('\n')
        file_results.write("gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.pvalue, universal epoch: "  + str(univ_epoch_curr) + ",  run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.pvalue))
        file_results.write('\n')
        
        print(time.time())
        
        ###########################
        
        #### 2-point probability function - mean of ALL
        
        #print("2-point probability function - mean of ALL ===========================================")
        
        two_pt_prob_mean_of_all = []
        
        for massaged_output in gen_samples:        
            ######################## Calculate 2-point probability function for this generated volume.
            volume_size = 64        # Volume is 64 voxels across.
            line_length = math.trunc(volume_size / 2) - 1
            
            sum_fn = 0
            for i in range(volume_size):  
                fn_goose = GooseEYE.S2((line_length, 1), massaged_output[i], massaged_output[i]) #31 is the distance over two points(white and black), (1, 31) gives us (x, y) different axis
                sum_fn += np.sum(fn_goose)
                len_fn_output = len(fn_goose)
                
            mean = (sum_fn/len_fn_output)/volume_size    

            two_pt_prob_mean_of_all.append(mean)
            #print(str(mean))
            ######################## Done calculating the 2-point probability function.
    
        kolmogorov_smirnov = ks_2samp(gt_2_pt_prob_mean_of_all, two_pt_prob_mean_of_all)
        KS_lineal_path_mean_of_all.append(kolmogorov_smirnov)
        
        #print("Lineal path function - mean of max (porosity), run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " ===========================================")
        #print(kolmogorov_smirnov)

        print("gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all, kolmogorov_smirnov.statistic, universal epoch: "  + str(univ_epoch_curr) + ",  run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.statistic))
        print("gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all, kolmogorov_smirnov.pvalue, universal epoch: "  + str(univ_epoch_curr) + ",  run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.pvalue))
        file_results.write("gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all, universal epoch: "  + str(univ_epoch_curr) + ",  kolmogorov_smirnov.statistic, run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.statistic))
        file_results.write('\n')
        file_results.write("gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all, universal epoch: "  + str(univ_epoch_curr) + ",  kolmogorov_smirnov.pvalue, run: "  + str(run_curr) + ", epoch: " + str(epoch_curr) + " = " + str(kolmogorov_smirnov.pvalue))
        file_results.write('\n')
    
        #input("go? ")
        
        
        file_results.flush()
    
    file_results.close()

elif req_option == 13:
    print("NOT IMPLEMENTED")
    pass

elif req_option == 14:
    print("*** Creating K/S graphs ***")
    
    outfile_loc ="C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/"
    model_loc = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/model_backup/"
    log_scale = False

    run_num_file = '1651647407'
    file_data_in = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/' + run_num_file + '_K-S_analysis.txt'

    timestampB = str(int(time.time()))

    file_out = os.path.join(outfile_loc, run_num_file + "_" + timestampB + "k-s_pvalue.png")
    
    f = open(file_data_in, 'r')
        
    univ_epoch = []
    ks_stat = []
    ks_pval = []
        
    for l in f:
        l = l.strip()
        if l.find('gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.pvalue,') == 0:
            dot_loc = l.rfind(".")
            print(dot_loc)
            numeric_value = float(l[(dot_loc-1):])
            ks_pval.append(numeric_value)
            
            univ_loc = l.find("universal epoch:")
            run_loc = l.rfind("run:")
            epoch_value = int(l[(univ_loc+16):(run_loc-3)])
            univ_epoch.append(epoch_value)
        
        
    print("Generating line graph...")
    plt.plot(univ_epoch, ks_pval, label="p-Value", marker='o', color='black')
    plt.ylabel("Kolmogorov/Smirnov p-Value")
    plt.xlabel("Epoch")
    
    if log_scale:
        plt.yscale('log')
    
    plt.title("Kolmogorov/Smirnov p-Value per Epoch")
    #plt.legend(loc="Loss")
    plt.savefig(file_out, dpi=300)
    print("Saved to " + file_out)
    plt.show()
    plt.clf()

    file_out = os.path.join(outfile_loc, run_num_file + "_" + timestampB + "k-s_statistic.png")
    
    f = open(file_data_in, 'r')
        
    univ_epoch = []
    ks_stat = []
    ks_pval = []
        
    for l in f:
        l = l.strip()
        if l.find('gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.statistic,') == 0:
            dot_loc = l.rfind(".")
            numeric_value = float(l[(dot_loc-1):])
            ks_pval.append(numeric_value)
            
            univ_loc = l.find("universal epoch:")
            run_loc = l.rfind("run:")
            epoch_value = int(l[(univ_loc+16):(run_loc-3)])
            univ_epoch.append(epoch_value)
        
        
    print("Generating line graph...")
    plt.plot(univ_epoch, ks_pval, label="Distance", marker='o', color='black')
    plt.ylabel("Kolmogorov/Smirnov Distance")
    plt.xlabel("Epoch")
    
    if log_scale:
        plt.yscale('log')
    
    plt.title("Kolmogorov/Smirnov Distance per Epoch")
    #plt.legend(loc="Loss")
    plt.savefig(file_out, dpi=300)
    print("Saved to " + file_out)
    plt.show()
    plt.clf()
    
    
elif req_option == 15:
    print("*** Creating K/S graphs with error bars ***")
    
    err_bar_width = 0.7
    err_bar_markersize = 2.5
    plot_aspect = 3.12 / 5.46
    plot_width = 10
    plot_height = plot_width * plot_aspect
    
    outfile_loc ="C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis/"
    image_output_loc = outfile_loc

    log_scale = False
    want_poly = True
    want_polyline = True

    run_num_files = ['1651656106', '1651898942', '1651899087']
    files_data_in = [outfile_loc + run_num_files[0] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[1] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[2] + '_K-S_analysis.txt']

    timestampC = str(int(time.time()))
    
    #####################################
    
    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_linealpath_k-s_statistic.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_linealpath_k-s_statistic.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.statistic,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-3)])
        
                #ks_stat_all.append(ks_stat)
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find lowest K/S value
    lowest_val = 1000000
    for i in range(len(ks_stat_error)):
        if ks_stat_error[i] <= lowest_val:
            low_epoch = univ_epoch[i]
            print("lowest epoch: " + str(low_epoch) + ", K/S value: " + str(ks_stat_error[i]))
            lowest_val = ks_stat_error[i]
            epoch_lowest = i

    # Now find a line that fits the points.
    X = np.array(univ_epoch).reshape(-1, 1) 
    Y = np.array(ks_stat_mean).reshape(-1, 1) 
    lin_reg = LinearRegression()  

    if want_poly:
        print("Polynomial")
        poly_obj = PolynomialFeatures(degree=3, include_bias=False)
        polyn_features = poly_obj.fit_transform(X)
        lin_reg.fit(polyn_features, Y)
        y_predict = lin_reg.predict(polyn_features)
    else:
        print("Purely linear")
        lin_reg.fit(X, Y) 
        y_predict = lin_reg.predict(X)
    
    min_val = 10000
    min_locs = []
    for i in range(len(univ_epoch)):
        curr_val = y_predict[i]
        if curr_val == min_val:
            min_locs.append(i)
        elif curr_val < min_val:
            min_val = curr_val
            min_locs = [i]  

    #print(y_predict)

    print("Lowest value in the trendline:")
    print(min_locs)
    print(min_val)
    print()
    print("Lowest val among the experimental data:")
    print(lowest_val)
    print("...in epoch: " + str(epoch_lowest))
    
    print("Generating line graph...")

    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)

    y_dir_err = [ks_stat_error, ks_stat_error]      # error is symmetric
    plt.errorbar(univ_epoch, ks_stat_mean, yerr=y_dir_err, c='black', ecolor='black', fmt=".", elinewidth=err_bar_width, markersize=err_bar_markersize)

    if want_polyline:
        plt.plot(X, y_predict, color='red')         # Trendline

    plt.ylabel("Kolmogorov/Smirnov Distance")
    plt.xlabel("Training Epoch")
    
    plt.ylim([0, 0.25])
    
    if log_scale:
        plt.yscale('log')
        
    plt.title("Kolmogorov/Smirnov Distance for the Mean of the Lineal Path")

    if want_polyline:
        plt.legend(['Regression Line', 'K/S Distance'], loc="upper right")
    else:
        plt.legend(['K/S Distance'], loc="upper right")

    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    #plt.show()
    plt.clf()    
    
    
    #####################################  2-pt below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_2ptprob_k-s_statistic.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_2ptprob_k-s_statistic.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all,') == 0 and l.find('kolmogorov_smirnov.statistic') != -1:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                str_part = l[univ_loc+16:]

                comma_loc = str_part.find(",")
                epoch_value = int( str_part[:comma_loc] )
        
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find lowest K/S value
    lowest_val = 1000000
    for i in range(len(ks_stat_error)):
        if ks_stat_error[i] <= lowest_val:
            low_epoch = univ_epoch[i]
            print("lowest epoch: " + str(low_epoch) + ", K/S value: " + str(ks_stat_error[i]))
            lowest_val = ks_stat_error[i]
            epoch_lowest = i

    # Now find a line that fits the points.
    X = np.array(univ_epoch).reshape(-1, 1) 
    Y = np.array(ks_stat_mean).reshape(-1, 1) 
    lin_reg = LinearRegression()  

    if want_poly:
        print("Polynomial")
        poly_obj = PolynomialFeatures(degree=3, include_bias=False)
        polyn_features = poly_obj.fit_transform(X)
        lin_reg.fit(polyn_features, Y)
        y_predict = lin_reg.predict(polyn_features)
    else:
        print("Purely linear")
        lin_reg.fit(X, Y) 
        y_predict = lin_reg.predict(X)
    
    min_val = 10000
    min_locs = []
    for i in range(len(univ_epoch)):
        curr_val = y_predict[i]
        if curr_val == min_val:
            min_locs.append(i)
        elif curr_val < min_val:
            min_val = curr_val
            min_locs = [i]  

    #print(y_predict)

    print("Lowest value in the trendline:")
    print(min_locs)
    print(min_val)
    print()
    print("Lowest val among the experimental data:")
    print(lowest_val)
    print("...in epoch: " + str(epoch_lowest))
    
    print("Generating line graph...")

    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)

    y_dir_err = [ks_stat_error, ks_stat_error]      # error is symmetric
    plt.errorbar(univ_epoch, ks_stat_mean, yerr=y_dir_err, c='black', ecolor='black', fmt=".", elinewidth=err_bar_width, markersize=err_bar_markersize)
    
    if want_polyline:
        plt.plot(X, y_predict, color='red')         # Trendline

    plt.ylabel("Kolmogorov/Smirnov Distance")
    plt.xlabel("Training Epoch")
    
    plt.ylim([0, 0.25])
    
    if log_scale:
        plt.yscale('log')
        
    plt.title("Kolmogorov/Smirnov Distance for the Mean of the Two-Point Probability Function")
    
    if want_polyline:
        plt.legend(['Regression Line', 'K/S Distance'], loc="upper right")
    else:
        plt.legend(['K/S Distance'], loc="upper right")
        
    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    #plt.show()
    plt.clf()    
    
    #####################################  porosity below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_porosity_k-s_statistic.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_porosity_k-s_statistic.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_porosity vs lineal_path_mean_of_max_data kolmogorov_smirnov.statistic,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-2)])
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find lowest K/S value
    lowest_val = 1000000
    for i in range(len(ks_stat_error)):
        if ks_stat_error[i] <= lowest_val:
            low_epoch = univ_epoch[i]
            print("lowest epoch: " + str(low_epoch) + ", K/S value: " + str(ks_stat_error[i]))
            lowest_val = ks_stat_error[i]
            epoch_lowest = i

    # Now find a line that fits the points.
    X = np.array(univ_epoch).reshape(-1, 1) 
    Y = np.array(ks_stat_mean).reshape(-1, 1) 
    lin_reg = LinearRegression()  

    if want_poly:
        print("Polynomial")
        poly_obj = PolynomialFeatures(degree=3, include_bias=False)
        polyn_features = poly_obj.fit_transform(X)
        lin_reg.fit(polyn_features, Y)
        y_predict = lin_reg.predict(polyn_features)
    else:
        print("Purely linear")
        lin_reg.fit(X, Y) 
        y_predict = lin_reg.predict(X)
    
    min_val = 10000
    min_locs = []
    for i in range(len(univ_epoch)):
        curr_val = y_predict[i]
        if curr_val == min_val:
            min_locs.append(i)
        elif curr_val < min_val:
            min_val = curr_val
            min_locs = [i]  

    #print(y_predict)

    print("Lowest value in the trendline:")
    print(min_locs)
    print(min_val)
    print()
    print("Lowest val among the experimental data:")
    print(lowest_val)
    print("...in epoch: " + str(epoch_lowest))
    
    print("Generating line graph...")

    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)

    y_dir_err = [ks_stat_error, ks_stat_error]      # error is symmetric
    #plt.plot(univ_epoch, ks_stat_mean, label="Distance", color='k', marker='o', linewidth=1, linestyle='dotted')
    plt.errorbar(univ_epoch, ks_stat_mean, yerr=y_dir_err, c='black', ecolor='black', fmt=".", elinewidth=err_bar_width, markersize=err_bar_markersize)
    
    if want_polyline:
        plt.plot(X, y_predict, color='red')         # Trendline
    
    plt.ylabel("Kolmogorov/Smirnov Distance")
    plt.xlabel("Training Epoch")
    
    plt.ylim([0, 0.25])
    
    if log_scale:
        plt.yscale('log')
        
    plt.title("Kolmogorov/Smirnov Distance for Porosity")

    if want_polyline:
        plt.legend(['Regression Line', 'K/S Distance'], loc="upper right")
    else:
        plt.legend(['K/S Distance'], loc="upper right")

    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    #plt.show()
    plt.clf()    
    
    print("If you want to see the trendline, set want_polyline to True and re-run option 15.")
    
    #####################################


elif req_option == 16:
    # Voxel display
    run_curr = '894172'
    epoch_curr = '180' 
    
    file_path_in_models = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/model_backup/"
    file_path_in_logs = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/out_files/'
    file_path_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/'
        
    model_file_name = str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.torchsave'

    print(model_file_name)
    
    model = DCGAN3D()
    checkpoint =  torch.load(os.path.join(file_path_in_models, model_file_name))
    model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
    model.eval() 
    
    # Make samples from generator
    continue_ans = "y"
    
    while continue_ans != "n":
        batch_size_loc = 1
        rand_input_raw = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
        rand_input = rand_input_raw.view(batch_size_loc, 64, 1, 1, 1)
        output = model.generator(rand_input)
        output = output.detach().numpy()
            
        # Force to exactly zero and exactly one so we have a binary volume (material and void).
        massaged_output = np.squeeze(output)
        massaged_output = np.around(massaged_output)
        
        print(rand_input_raw)
        
        out_str = ""
        for i in range(len(rand_input_raw)-1):
            out_str += str(rand_input_raw[i].item()) + ", "
        out_str += str(rand_input_raw[len(rand_input_raw)-1].item())
        print(out_str)
        
        # Display as an image
        z,x,y = massaged_output.nonzero()    
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(massaged_output, facecolors='silver', edgecolor='k')
        plt.show()
        
        continue_ans = input("Continue? (y/n) ")
    
elif req_option == 17:
    # Voxel display
    run_curr = '894172'
    epoch_curr = '180' 
    
    file_path_in_models = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/model_backup/"
    file_path_in_logs = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/out_files/'
    file_path_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/'
        
    model_file_name = str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.torchsave'

    print(model_file_name)
    
    model = DCGAN3D()
    checkpoint =  torch.load(os.path.join(file_path_in_models, model_file_name))
    model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
    model.eval() 
    
    # Make samples from generator
    continue_ans = "y"
    
    while continue_ans != "n":
        in_noise = input("Input your noise, comma-delimited: ")
        noise_items = in_noise.split(",")
        noise_data = np.zeros(64, dtype=np.single)
        
        for i in range(len(noise_items)):
            noise_data[i] += float(noise_items[i])
            
        print(noise_data)
            
        np_rand_input = np.array(noise_data)
        rand_input_raw = torch.from_numpy(np_rand_input)

        batch_size_loc = 1
        rand_input = rand_input_raw.view(batch_size_loc, 64, 1, 1, 1)
        output = model.generator(rand_input)
        output = output.detach().numpy()
        
        # Force to exactly zero and exactly one so we have a binary volume (material and void).
        massaged_output = np.squeeze(output)
        massaged_output = np.around(massaged_output)
        
        print(rand_input_raw)
        
        out_str = ""
        for i in range(len(rand_input_raw)-1):
            out_str += str(rand_input_raw[i].item()) + ", "
        out_str += str(rand_input_raw[len(rand_input_raw)-1].item())
        print(out_str)
        
        # Display as an image
        z,x,y = massaged_output.nonzero()    
        ax = plt.figure().add_subplot(projection='3d')
        plt.grid(False)
        plt.axis('off')
        ax.voxels(massaged_output, facecolors='oldlace', edgecolor='k')
        plt.show()
        
        continue_ans = input("Continue? (y/n) ")    

elif req_option == 18:
    model_location = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/models(torchsave)/'
    outfile_location = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/out_files/'
    image_output_loc = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/one_noise_all_models/'
    start_run_num = 999999
    
    timestampD = str(int(time.time()))

    #in_noise = input("Input your noise, comma-delimited: ")
    
    in_noise = "-0.08800923079252243, 0.39580321311950684, -0.8052524328231812, -0.13871890306472778, -0.29372555017471313, 0.2923274040222168, -2.1221213340759277, 0.855306088924408, 0.10293901711702347, -0.6732857823371887, -0.2714861333370209, 0.7865220308303833, -0.8273285031318665, 2.9451072216033936, 0.5558481812477112, 0.9901391863822937, 1.3214446306228638, 0.5277003049850464, -0.6072021126747131, 1.1921862363815308, 0.26992470026016235, 0.5514655113220215, -1.710094690322876, -1.1839689016342163, -0.031422536820173264, -0.5944361686706543, 0.7373148202896118, -0.2982260286808014, 1.6223398447036743, 2.161883592605591, 0.036563895642757416, -1.497494101524353, -2.17046856880188, -0.43129369616508484, -0.32606589794158936, -1.42513906955719, 3.011629819869995, -0.5819171071052551, -0.7875000834465027, 0.015368077903985977, -2.8817062377929688, -0.47058701515197754, -0.4655972421169281, -0.6610618829727173, -0.4476863741874695, 0.48865944147109985, 1.16494882106781, -0.5686888098716736, 1.4584836959838867, 1.3600279092788696, 0.6557131409645081, 0.3703085482120514, -0.9195609092712402, -0.40661919116973877, -0.020623479038476944, -0.7490653991699219, 1.3085993528366089, 1.3781172037124634, -0.4045710563659668, -0.3121279776096344, -0.5501620769500732, -0.8497989177703857, -0.8546901941299438, 1.2750352621078491"
    
    noise_items = in_noise.split(",")
    noise_data = np.zeros(64, dtype=np.single)
    
    for i in range(len(noise_items)):
        noise_data[i] += float(noise_items[i])
        
    print(noise_data)

    np_rand_input = np.array(noise_data)
    rand_input_raw = torch.from_numpy(np_rand_input)

    batch_size_loc = 1
    rand_input = rand_input_raw.view(batch_size_loc, 64, 1, 1, 1)

    
    saved_models_data = get_lists_of_saved_models(model_location, outfile_location, start_run_num)
    run_numbers = saved_models_data[0]
    epoch_numbers = saved_models_data[1]
    run_epoch_numbers = saved_models_data[2]
    
    incr = 0    
    
    nowww = time.time()
    
    for i in range(len(run_epoch_numbers)):
        print("Wallclock seconds so far: " + str(time.time() - nowww))
        
        run_curr = run_numbers[i]
        epoch_curr = run_epoch_numbers[i]
        
        model_file_name = str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.torchsave'
        
        print(model_file_name)
        
        model = DCGAN3D()
        checkpoint =  torch.load(os.path.join(model_location, model_file_name))
        model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
        model.eval() 
            
        output = model.generator(rand_input)
        output = output.detach().numpy()
        
        # Force to exactly zero and exactly one so we have a binary volume (material and void).
        massaged_output = np.squeeze(output)
        massaged_output = np.around(massaged_output)
        
        # Display as an image
        z,x,y = massaged_output.nonzero()    
        ax = plt.figure().add_subplot(projection='3d')
        plt.grid(False)
        plt.axis('off')
        ax.voxels(massaged_output, facecolors='oldlace', edgecolor='k', linewidth=0.2)
        #plt.show()
        
        file_name_img = timestampD + '_' + "{:0>4d}".format(incr) + "_" + str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.png'
        plt.savefig(os.path.join(image_output_loc, file_name_img), dpi=300)
        plt.close()
        
        incr += 1
        
        #input("Go? ")

    #####################################

elif req_option == 19:
    model_location = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/models(torchsave)/'
    image_output_loc = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/one_model_sliding_noise/'
    start_run_num = 999999
    
    wanted_mutations = 100
    distance_to_move = 0.1
    element_to_change = 6
    
    timestampD = str(int(time.time()))

    nowww = time.time()

    print("Wallclock seconds so far: " + str(time.time() - nowww))

    #in_noise = input("Input your noise, comma-delimited: ")
    
    in_noise = "-0.08800923079252243, 0.39580321311950684, -0.8052524328231812, -0.13871890306472778, -0.29372555017471313, 0.2923274040222168, -2.1221213340759277, 0.855306088924408, 0.10293901711702347, -0.6732857823371887, -0.2714861333370209, 0.7865220308303833, -0.8273285031318665, 2.9451072216033936, 0.5558481812477112, 0.9901391863822937, 1.3214446306228638, 0.5277003049850464, -0.6072021126747131, 1.1921862363815308, 0.26992470026016235, 0.5514655113220215, -1.710094690322876, -1.1839689016342163, -0.031422536820173264, -0.5944361686706543, 0.7373148202896118, -0.2982260286808014, 1.6223398447036743, 2.161883592605591, 0.036563895642757416, -1.497494101524353, -2.17046856880188, -0.43129369616508484, -0.32606589794158936, -1.42513906955719, 3.011629819869995, -0.5819171071052551, -0.7875000834465027, 0.015368077903985977, -2.8817062377929688, -0.47058701515197754, -0.4655972421169281, -0.6610618829727173, -0.4476863741874695, 0.48865944147109985, 1.16494882106781, -0.5686888098716736, 1.4584836959838867, 1.3600279092788696, 0.6557131409645081, 0.3703085482120514, -0.9195609092712402, -0.40661919116973877, -0.020623479038476944, -0.7490653991699219, 1.3085993528366089, 1.3781172037124634, -0.4045710563659668, -0.3121279776096344, -0.5501620769500732, -0.8497989177703857, -0.8546901941299438, 1.2750352621078491"
    run_curr = 894172
    epoch_curr = 180
    
    noise_items = in_noise.split(",")
    noise_data = np.zeros(64, dtype=np.single)
    
    for i in range(len(noise_items)):
        noise_data[i] += float(noise_items[i])
        
    print(noise_data)

    model_file_name = str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.torchsave'
    
    print(model_file_name)
    
    model = DCGAN3D()
    checkpoint =  torch.load(os.path.join(model_location, model_file_name))
    model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
    model.eval() 
    
    incr = 0
    
    for i in range(wanted_mutations):
        print("Wallclock seconds start of loop: " + str(time.time() - nowww))
        
        np_rand_input = np.array(noise_data)
        rand_input_raw = torch.from_numpy(np_rand_input)
    
        batch_size_loc = 1
        rand_input = rand_input_raw.view(batch_size_loc, 64, 1, 1, 1)
                
        output = model.generator(rand_input)
        output = output.detach().numpy()
        
        # Force to exactly zero and exactly one so we have a binary volume (material and void).
        massaged_output = np.squeeze(output)
        massaged_output = np.around(massaged_output)
        
        print("Wallclock seconds after generating volume: " + str(time.time() - nowww))
        
        #print(rand_input_raw)
        
        # Display as an image
        z,x,y = massaged_output.nonzero()    
        ax = plt.figure().add_subplot(projection='3d')
        plt.grid(False)
        plt.axis('off')
        ax.voxels(massaged_output, facecolors='oldlace', edgecolor='k', linewidth=0.2)
        #plt.show()
        
        print("Wallclock seconds after making voxel image: " + str(time.time() - nowww))
        
        file_name_img = timestampD + '_' + "{:0>4d}".format(incr) + "_" + str(run_curr) + '_model_and_opt_save_' + str(epoch_curr) +'.png'
        #plt.savefig(os.path.join(image_output_loc, file_name_img), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(image_output_loc, file_name_img), dpi=300)
        plt.close()
        
        print("Wallclock seconds after saving voxel image: " + str(time.time() - nowww))
        
        noise_data[element_to_change] += distance_to_move
        
        incr += 1

if req_option == 20:
    file_path_out_loc = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/256_box_with_slices/'
    
    file_in = os.path.join(file_path_in, '10_03_256.mat')
    file_out = os.path.join(file_path_out_loc, timestamp1 + '_256_with_division_planes.png')
    alpha_const = 0.05
    
    print("Printing 256-volume with planes.  Input should be at " + file_in)
    print("Generating image...")
    
    nowww = time.time()
    
    incoming_data = loadmat(file_in)['bin']
    
    # Break up
    divisionplanes = 4
    sizeofgap = 10
    additionalneeded = divisionplanes * sizeofgap
    sidelen = len(incoming_data[0])
    newside = sidelen + additionalneeded
    subside = sidelen / divisionplanes
    
    print("subside: " + str(subside))
    
    newdata = np.zeros((newside, newside, newside))
    
    for x in range(sidelen):
        #print("x is now " + str(x))
        for y in range(sidelen):
            for z in range(sidelen):
                boxespassed_x = math.trunc(x / subside)
                boxespassed_y = math.trunc(y / subside)
                boxespassed_z = math.trunc(z / subside)
                
                newx = int(boxespassed_x * sizeofgap) + x
                newy = int(boxespassed_y * sizeofgap) + y
                newz = int(boxespassed_z * sizeofgap) + z
                
                newdata[newx,newy,newz] = incoming_data[x,y,z]
    
    print("Wallclock seconds after making new output: " + str(time.time() - nowww))

    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')
    #plt.show()
 
    print("Wallclock seconds after readying plot: " + str(time.time() - nowww))
    
    ax.voxels(newdata, facecolors='oldlace', edgecolor='k', linewidth=0.01)

    print("Wallclock seconds after making voxel image: " + str(time.time() - nowww))
    
    plt.savefig(file_out, dpi=300)
    plt.close()
    
    print("Wallclock seconds after done: " + str(time.time() - nowww))
    
    print("Figure saved to " + file_out)
    
    ###########################################################################################

if req_option == 21:
    file_path_out_loc = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/256_box_with_slices/'
    
    file_in = os.path.join(file_path_in, '10_03_256.mat')
    file_out = os.path.join(file_path_out_loc, timestamp1 + '_256_uncut.png')
    alpha_const = 0.05
    
    print("Printing 256-volume uncut.  Input should be at " + file_in)
    print("Generating image...")
    
    nowww = time.time()
    
    incoming_data = loadmat(file_in)['bin']
    
    print("Wallclock seconds after opening file: " + str(time.time() - nowww))

    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')
    #plt.show()
 
    print("Wallclock seconds after readying plot: " + str(time.time() - nowww))
    
    ax.voxels(incoming_data, facecolors='oldlace', edgecolor='k', linewidth=0.01)

    print("Wallclock seconds after making voxel image: " + str(time.time() - nowww))
    
    plt.savefig(file_out, dpi=300)
    plt.close()
    
    print("Wallclock seconds after done: " + str(time.time() - nowww))
    
    print("Figure saved to " + file_out)
    
    ###########################################################################################

if req_option == 22:
    file_path_out_loc = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/256_box_with_slices/'
    
    file_in = os.path.join(file_path_in, '10_03_256.mat')
    file_out = os.path.join(file_path_out_loc, timestamp1 + '_64_subvolume.png')
    alpha_const = 0.05
    
    print("Printing on 64-volume.  Input should be at " + file_in)
    print("Generating image...")
    
    nowww = time.time()
    
    incoming_data = loadmat(file_in)['bin']
    
    incoming_data = incoming_data[:64, :64, :64]
    
    print("Wallclock seconds after opening file: " + str(time.time() - nowww))

    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')
 
    print("Wallclock seconds after readying plot: " + str(time.time() - nowww))
    
    ax.voxels(incoming_data, facecolors='oldlace', edgecolor='k', linewidth=0.01)

    print("Wallclock seconds after making voxel image: " + str(time.time() - nowww))
    
    plt.savefig(file_out, dpi=300)
    plt.close()
    
    print("Wallclock seconds after done: " + str(time.time() - nowww))
    
    print("Figure saved to " + file_out)
    
    ###########################################################################################

elif req_option == 23:
    print("*** Creating K/S p-value graphs with error bars ***")
    
    outfile_loc ="C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis/"
    image_output_loc = outfile_loc

    log_scale = False
    want_poly = True
    want_polyline = True
    
    plot_aspect = 3.12 / 5.46
    plot_width = 10
    plot_height = plot_width * plot_aspect    

    run_num_files = ['1651656106', '1651898942', '1651899087']
    files_data_in = [outfile_loc + run_num_files[0] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[1] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[2] + '_K-S_analysis.txt']

    timestampC = str(int(time.time()))
    
    #####################################
    
    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_linealpath_k-s_statistic_p-value.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_linealpath_k-s_statistic_p-value.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.pvalue,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-3)])
        
                #ks_stat_all.append(ks_stat)
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find lowest K/S value
    lowest_val = 1000000
    for i in range(len(ks_stat_error)):
        if ks_stat_error[i] <= lowest_val:
            low_epoch = univ_epoch[i]
            print("lowest epoch: " + str(low_epoch) + ", K/S value: " + str(ks_stat_error[i]))
            lowest_val = ks_stat_error[i]
            epoch_lowest = i

    # Now find a line that fits the points.
    X = np.array(univ_epoch).reshape(-1, 1) 
    Y = np.array(ks_stat_mean).reshape(-1, 1) 
    lin_reg = LinearRegression()  

    if want_poly:
        print("Polynomial")
        poly_obj = PolynomialFeatures(degree=3, include_bias=False)
        polyn_features = poly_obj.fit_transform(X)
        lin_reg.fit(polyn_features, Y)
        y_predict = lin_reg.predict(polyn_features)
    else:
        print("Purely linear")
        lin_reg.fit(X, Y) 
        y_predict = lin_reg.predict(X)
    
    min_val = 10000
    min_locs = []
    for i in range(len(univ_epoch)):
        curr_val = y_predict[i]
        if curr_val == min_val:
            min_locs.append(i)
        elif curr_val < min_val:
            min_val = curr_val
            min_locs = [i]  

    #print(y_predict)

    print("Lowest value in the trendline:")
    print(min_locs)
    print(min_val)
    print()
    print("Lowest val among the experimental data:")
    print(lowest_val)
    print("...in epoch: " + str(epoch_lowest))
    
    print("Generating line graph...")

    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)

    y_dir_err = [ks_stat_error, ks_stat_error]      # error is symmetric
    plt.errorbar(univ_epoch, ks_stat_mean, yerr=y_dir_err, c='black', ecolor='black', fmt=".", elinewidth=1)
    
    if want_polyline:
        plt.plot(X, y_predict, color='red')         # Trendline
        plt.axhline(y=0.05, linestyle='dotted')

    plt.ylabel("p-Value for the Kolmogorov/Smirnov Distance")
    plt.xlabel("Training Epoch")
    
    #plt.ylim([0, 0.25])
    
    if log_scale:
        plt.yscale('log')
        
    plt.title("p-Value for the Kolmogorov/Smirnov for the Mean of the Lineal Path")

    if want_polyline:
        plt.legend(['Regression Line', 'Reject Null Hypothesis', 'p-Value'], loc="upper right")
    else:
        plt.legend(['K/S Distance'], loc="upper right")

    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    plt.show()
    plt.clf()    
    
    
    #####################################  2-pt below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_2ptprob_k-s_statistic_p-value.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_2ptprob_k-s_statistic_p-value.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all,') == 0 and l.find('kolmogorov_smirnov.pvalue') != -1:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                str_part = l[univ_loc+16:]

                comma_loc = str_part.find(",")
                epoch_value = int( str_part[:comma_loc] )
        
                #ks_stat_all.append(ks_stat)
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        #ks_stat_mean.append( (ks_stat_all[0][i] + ks_stat_all[1][i] + ks_stat_all[2][i])/3 )
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find lowest K/S value
    lowest_val = 1000000
    for i in range(len(ks_stat_error)):
        if ks_stat_error[i] <= lowest_val:
            low_epoch = univ_epoch[i]
            print("lowest epoch: " + str(low_epoch) + ", K/S value: " + str(ks_stat_error[i]))
            lowest_val = ks_stat_error[i]
            epoch_lowest = i

    # Now find a line that fits the points.
    X = np.array(univ_epoch).reshape(-1, 1) 
    Y = np.array(ks_stat_mean).reshape(-1, 1) 
    lin_reg = LinearRegression()  

    if want_poly:
        print("Polynomial")
        poly_obj = PolynomialFeatures(degree=3, include_bias=False)
        polyn_features = poly_obj.fit_transform(X)
        lin_reg.fit(polyn_features, Y)
        y_predict = lin_reg.predict(polyn_features)
    else:
        print("Purely linear")
        lin_reg.fit(X, Y) 
        y_predict = lin_reg.predict(X)
    
    min_val = 10000
    min_locs = []
    for i in range(len(univ_epoch)):
        curr_val = y_predict[i]
        if curr_val == min_val:
            min_locs.append(i)
        elif curr_val < min_val:
            min_val = curr_val
            min_locs = [i]  

    print("Lowest value in the trendline:")
    print(min_locs)
    print(min_val)
    print()
    print("Lowest val among the experimental data:")
    print(lowest_val)
    print("...in epoch: " + str(epoch_lowest))
    
    print("Generating line graph...")
    #plt.axvline(x=min_locs[0], linestyle='--')                  # Best point on the trendline
    #plt.axvline(epoch_lowest, linestyle='dotted')                  # Best point among the experimental data

    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)

    y_dir_err = [ks_stat_error, ks_stat_error]      # error is symmetric
    plt.errorbar(univ_epoch, ks_stat_mean, yerr=y_dir_err, c='black', ecolor='black', fmt=".", elinewidth=1)
    
    if want_polyline:
        plt.plot(X, y_predict, color='red')         # Trendline
        plt.axhline(y=0.05, linestyle='dotted')
    
    plt.ylabel("p-Value for the Kolmogorov/Smirnov Distance")
    plt.xlabel("Training Epoch")
    
    #plt.ylim([0, 0.25])
    
    if log_scale:
        plt.yscale('log')
        
    plt.title("p-value for the Kolmogorov/Smirnov Distance for the Mean of the Two-Point Probability Function")
    
    if want_polyline:
        plt.legend(['Regression Line', 'Reject Null Hypothesis', 'p-Value'], loc="upper right")
    else:
        plt.legend(['p-value'], loc="upper right")
        
    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    plt.show()
    plt.clf()    
    
    #####################################  porosity below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_porosity_k-s_statistic_p-value.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_porosity_k-s_statistic_p-value.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_porosity vs lineal_path_mean_of_max_data kolmogorov_smirnov.pvalue,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-2)])
        
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find lowest K/S value
    lowest_val = 1000000
    for i in range(len(ks_stat_error)):
        if ks_stat_error[i] <= lowest_val:
            low_epoch = univ_epoch[i]
            print("lowest epoch: " + str(low_epoch) + ", K/S value: " + str(ks_stat_error[i]))
            lowest_val = ks_stat_error[i]
            epoch_lowest = i

    # Now find a line that fits the points.
    X = np.array(univ_epoch).reshape(-1, 1) 
    Y = np.array(ks_stat_mean).reshape(-1, 1) 
    lin_reg = LinearRegression()  

    if want_poly:
        print("Polynomial")
        poly_obj = PolynomialFeatures(degree=3, include_bias=False)
        polyn_features = poly_obj.fit_transform(X)
        lin_reg.fit(polyn_features, Y)
        y_predict = lin_reg.predict(polyn_features)
    else:
        print("Purely linear")
        lin_reg.fit(X, Y) 
        y_predict = lin_reg.predict(X)
    
    min_val = 10000
    min_locs = []
    for i in range(len(univ_epoch)):
        curr_val = y_predict[i]
        if curr_val == min_val:
            min_locs.append(i)
        elif curr_val < min_val:
            min_val = curr_val
            min_locs = [i]  

    #print(y_predict)

    print("Lowest value in the trendline:")
    print(min_locs)
    print(min_val)
    print()
    print("Lowest val among the experimental data:")
    print(lowest_val)
    print("...in epoch: " + str(epoch_lowest))
    
    print("Generating line graph...")

    fig, ax = plt.subplots()
    fig.set_size_inches(plot_width, plot_height)

    y_dir_err = [ks_stat_error, ks_stat_error]      # error is symmetric
    #plt.plot(univ_epoch, ks_stat_mean, label="Distance", color='k', marker='o', linewidth=1, linestyle='dotted')
    plt.errorbar(univ_epoch, ks_stat_mean, yerr=y_dir_err, c='black', ecolor='black', fmt=".", elinewidth=1)
    
    if want_polyline:
        plt.plot(X, y_predict, color='red')         # Trendline
        plt.axhline(y=0.05, linestyle='dotted')
    
    plt.ylabel("p-Value for the Kolmogorov/Smirnov Distance")
    plt.xlabel("Training Epoch")
    
    #plt.ylim([0, 0.25])
    
    if log_scale:
        plt.yscale('log')
        
    plt.title("p-Value for the Kolmogorov/Smirnov Distance for the Porosity")

    if want_polyline:
        plt.legend(['Regression Line', 'Reject Null Hypothesis', 'p-Value'], loc="upper right")
    else:
        plt.legend(['p-Value'], loc="upper right")

    plt.savefig(file_out, dpi=1200)
    print("Saved to " + file_out)
    plt.show()
    plt.clf()    
    
    
    #####################################

elif req_option == 24:
    print("*** Finding best model: p-values ***")
    
    outfile_loc ="C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis/"
    image_output_loc = outfile_loc

    log_scale = False
    want_poly = True
    want_polyline = True

    run_num_files = ['1651656106', '1651898942', '1651899087']
    files_data_in = [outfile_loc + run_num_files[0] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[1] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[2] + '_K-S_analysis.txt']

    timestampC = str(int(time.time()))
    
    best_lineal = []
    best_2pt = []
    best_porosity = []
    
    #####################################
    
    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_linealpath_k-s_statistic_p-value.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_linealpath_k-s_statistic_p-value.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.pvalue,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-3)])
        
                #ks_stat_all.append(ks_stat)
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find p-values above 0.05
    print("Lineal path")
    for i in range(len(ks_stat_error)):
        if ks_stat_mean[i] > 0.05:
            print("epoch " + str(univ_epoch[i]) + ", p-value: " + str(ks_stat_mean[i]))
            best_lineal.append(univ_epoch[i])


    
    #####################################  2-pt below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_2ptprob_k-s_statistic_p-value.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_2ptprob_k-s_statistic_p-value.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all,') == 0 and l.find('kolmogorov_smirnov.pvalue') != -1:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                str_part = l[univ_loc+16:]

                comma_loc = str_part.find(",")
                epoch_value = int( str_part[:comma_loc] )
        
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find p-values above 0.05
    print("2-pt probability")
    for i in range(len(ks_stat_mean)):
        if ks_stat_mean[i] > 0.05:
            print("epoch " + str(univ_epoch[i]) + ", p-value: " + str(ks_stat_mean[i]))
            best_2pt.append(univ_epoch[i])


    
    #####################################  porosity below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_porosity_k-s_statistic_p-value.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_porosity_k-s_statistic_p-value.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_porosity vs lineal_path_mean_of_max_data kolmogorov_smirnov.pvalue,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-2)])
        
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    # Find p-values above 0.05
    print("Porosity")
    for i in range(len(ks_stat_mean)):
        if ks_stat_mean[i] > 0.05:
            print("epoch " + str(univ_epoch[i]) + ", p-value: " + str(ks_stat_mean[i]))
            best_porosity.append(univ_epoch[i])
    
    print(best_lineal)
    print("***************")
    print(best_2pt)
    print("***************")
    print(best_porosity)
    
    #####################################

elif req_option == 25:
    print("*** Selecting model: best K/S distances ***")
    
    outfile_loc ="C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis/"
    image_output_loc = outfile_loc

    log_scale = False
    want_poly = True
    want_polyline = False

    run_num_files = ['1651656106', '1651898942', '1651899087']
    files_data_in = [outfile_loc + run_num_files[0] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[1] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[2] + '_K-S_analysis.txt']

    timestampC = str(int(time.time()))
    
    #####################################
    
    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_linealpath_k-s_statistic.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_linealpath_k-s_statistic.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_lineal_path_mean_of_all vs lineal_path_mean_of_all, kolmogorov_smirnov.statistic,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-3)])
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)


    best_distances = [999999,999999,999999,999999,999999,999999]
    best_distance_epochs = [0,0,0,0,0,0]
    
    for i in range(len(univ_epoch)):
        for x in range(len(best_distances)):
            if ks_stat_mean[i] < best_distances[x]:
                best_distances[x] = ks_stat_mean[i]
                best_distance_epochs[x] = univ_epoch[i]
                break
        
    print("Lineal path mean, statistic")    
    print(best_distances)        
    print(best_distance_epochs)
       
    best_distances = []
    best_distance_epochs = []
    
    #####################################  2-pt below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_2ptprob_k-s_statistic.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_2ptprob_k-s_statistic.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all,') == 0 and l.find('kolmogorov_smirnov.statistic') != -1:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                str_part = l[univ_loc+16:]

                comma_loc = str_part.find(",")
                epoch_value = int( str_part[:comma_loc] )
        
                #ks_stat_all.append(ks_stat)
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    best_distances = [999999,999999,999999,999999,999999,999999]
    best_distance_epochs = [0,0,0,0,0,0]
    
    for i in range(len(univ_epoch)):
        for x in range(len(best_distances)):
            if ks_stat_mean[i] < best_distances[x]:
                best_distances[x] = ks_stat_mean[i]
                best_distance_epochs[x] = univ_epoch[i]
                break
        
    print("2-point prob fn, statistic")    
    print(best_distances)        
    print(best_distance_epochs)
    
    best_distances = []
    best_distance_epochs = []

    
    #####################################  porosity below

    if want_polyline:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_POLYLINE_porosity_k-s_statistic.png")
    else:
        file_out = os.path.join(image_output_loc, timestampC + "_runs_" + run_num_files[0] + "_" + run_num_files[1] + "_" + run_num_files[2] + "_porosity_k-s_statistic.png")
    
    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    best_distances = []
    best_distance_epochs = []
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_porosity vs lineal_path_mean_of_max_data kolmogorov_smirnov.statistic,') == 0:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                run_loc = l.rfind("run:")
                epoch_value = int(l[(univ_loc+16):(run_loc-2)])
                
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        #ks_stat_mean.append( (ks_stat_all[0][i] + ks_stat_all[1][i] + ks_stat_all[2][i])/3 )
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    best_distances = [999999,999999,999999,999999,999999,999999]
    best_distance_epochs = [0,0,0,0,0,0]
    
    for i in range(len(univ_epoch)):
        for x in range(len(best_distances)):
            if ks_stat_mean[i] < best_distances[x]:
                best_distances[x] = ks_stat_mean[i]
                best_distance_epochs[x] = univ_epoch[i]
                break
        
    print("2-point prob fn, statistic")    
    print(best_distances)        
    print(best_distance_epochs)


    
    #####################################

elif req_option == 26:
    print("*** Getting some k/s info for epoch 38 ***")
    
    outfile_loc ="C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis/"
    image_output_loc = outfile_loc

    log_scale = False
    want_poly = True
    want_polyline = False

    run_num_files = ['1651656106', '1651898942', '1651899087']
    files_data_in = [outfile_loc + run_num_files[0] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[1] + '_K-S_analysis.txt',
                        outfile_loc + run_num_files[2] + '_K-S_analysis.txt']

    timestampC = str(int(time.time()))

    best_distances = []
    best_distance_epochs = []
    
    #####################################  2-pt below

    ks_stat_all = []
    univ_epoch = []
    
    first_run = True
    
    for file_data_in in files_data_in:
        f = open(file_data_in, 'r')
        
        ks_stat = []
            
        for l in f:
            l = l.strip()
            if l.find('gt_2_pt_prob_mean_of_all vs two_pt_prob_mean_of_all,') == 0 and l.find('kolmogorov_smirnov.statistic') != -1:
                dot_loc = l.rfind(".")
                numeric_value = float(l[(dot_loc-1):])
                ks_stat.append(numeric_value)
                
                univ_loc = l.find("universal epoch:")
                str_part = l[univ_loc+16:]

                comma_loc = str_part.find(",")
                epoch_value = int( str_part[:comma_loc] )
        
                if first_run:
                    univ_epoch.append(epoch_value)

        ks_stat_all.append(ks_stat)
        
        first_run = False

    print("Calculating mean and error...")
    ks_stat_mean = []
    ks_stat_error = []
    
    for i in range(len(univ_epoch)):
        curr_data = [ks_stat_all[0][i], ks_stat_all[1][i], ks_stat_all[2][i] ]
        ks_stat_mean.append(sum(curr_data)/3.0)
        std_er = stats.sem(curr_data, axis=None, ddof=0)
        ks_stat_error.append(std_er)

    for i in range(32,44):
        for z in range(len(univ_epoch)):
            if univ_epoch[z] == i:
                print("2-pt prob, epoch: " + str(univ_epoch[z]) + ", val: " + str(ks_stat_mean[z]))

elif req_option == 27:
    print("*** Visualizing outputs from epoch 38 ***")
    
    model_found_at = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/predictions/models(torchsave)/880890_model_and_opt_save_38.torchsave'
    files_saved_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/best_model_outputs/'
    num_samples = 100
    
    model = DCGAN3D()
    checkpoint =  torch.load(model_found_at)
    model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
    model.eval() 
    
    for i in range(num_samples):
        batch_size_loc = 1
        raw_rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
        rand_input = raw_rand_input.view(batch_size_loc, 64, 1, 1, 1)
        output = model.generator(rand_input)
        output = output.detach().numpy()
            
        ## Save off the generated volume
        #base_vol_filename = run_num + '_generatedvolume_' + epoch_num + '_' + timestamp1 +  '_' + str(i)
        #file_name_vol = base_vol_filename + '.npy'
        #np.save(os.path.join(file_path_generated_volume, file_name_vol), output)
    
        # Force to exactly zero and exactly one so we have a binary volume (material and void).
        massaged_output = np.squeeze(output)
        massaged_output = np.around(massaged_output)
        
        ax = plt.figure().add_subplot(projection='3d')
        plt.grid(False)
        plt.axis('off')
        ax.voxels(massaged_output, facecolors='oldlace', edgecolor='k', linewidth=0.01)
    
        file_name_voxels = '880890_38_' + str(i) + '.png'
        file_out  = os.path.join(files_saved_out, file_name_voxels)
        
        plt.savefig(file_out, dpi=300)
        plt.close()

elif req_option == 28:
    print("*** Visualizing outputs from ground truth ***")
    
    file_in = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/test_samples/374_01_06_256.mat'
    files_saved_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/ground_truth_outputs/'
    num_samples = 10
    
    incoming_data = loadmat(file_in)['bin']

    incoming_data_chunks = []
    
    newchunk = incoming_data[:64,:64,:64]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[64:128,:64,:64]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[:64,64:128,:64]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[:64,:64, 64:128]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[192:,:64,:64]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[:64,192:,64:128]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[64:128,:64,64:128]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[128:192,128:192,128:192]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[:64,128:192,:64]
    incoming_data_chunks.append(newchunk)

    newchunk = incoming_data[:64,:64,128:192]
    incoming_data_chunks.append(newchunk)

    incr = 0
    for i in incoming_data_chunks:
        ax = plt.figure().add_subplot(projection='3d')
        plt.grid(False)
        plt.axis('off')
        ax.voxels(i, facecolors='oldlace', edgecolor='k', linewidth=0.01)
    
        file_name_voxels = '374_01_06_256_subvolume_' + str(incr) + '.png'
        file_out  = os.path.join(files_saved_out, file_name_voxels)
        
        plt.savefig(file_out, dpi=300)
        plt.close()
        
        incr += 1
        
elif req_option == 30:
    print("*** Making analysis data, starting with a saved model and the held-out test data. ***")
    
    model_fl = "880890_model_and_opt_save_38.torchsave"
    model_in  = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_in/" + model_fl
    ground_truth_in = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/test_samples/"
    info_out_file = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis_of_model_38/" + timestamp1 + "_model_38_numerical_data.txt"
    
    model_lineal_path_vals = []
    model_two_point_vals = []
    model_porosity_vals = []
    
    test_lineal_path_vals = []
    test_two_point_vals = []
    test_porosity_vals = []    
    
    print("Testing samples from the held-out data.")
    ## Data for the ground truth (test data)
    test_data_list = get_ground_truth_subvolumes(ground_truth_in)
    num_samples = len(test_data_list)
    print(">>> There are " + str(num_samples) + " subvolumes in the held-out data.")
    
    for s in test_data_list:
        test_lineal_path_vals.append(lineal_path_mean_all(s))
        test_two_point_vals.append(two_pt_prob_mean_all(s))
        test_porosity_vals.append(porosity_via_lineal_path(s))
    
    test_data_list = []
    
    print("Now testing samples generated by the model in " + model_in)
    #generated_samples = []
    
    model = DCGAN3D()
    checkpoint =  torch.load(model_in)
    model.load_state_dict(checkpoint['model_state_dict'])      # https://stackoverflow.com/questions/51811154/load-a-pretrained-model-pytorch-dict-object-has-no-attribute-eval
    model.eval()
    
    ### Data for the model.
    for i in range(num_samples):
        batch_size_loc = 1
        rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
        rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
        output = model.generator(rand_input)
        output = output.detach().numpy()
        
        model_lineal_path_vals.append(lineal_path_mean_all(output))
        
    for i in range(num_samples):
        batch_size_loc = 1
        rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
        rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
        output = model.generator(rand_input)
        output = output.detach().numpy()
        
        model_two_point_vals.append(two_pt_prob_mean_all(output))

    # Last, porosity: Mean of the max of either lineal path or 2-point prob; we'll use linear path
    for i in range(num_samples):
        batch_size_loc = 1
        rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
        rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
        output = model.generator(rand_input)
        output = output.detach().numpy()
        
        model_porosity_vals.append(porosity_via_lineal_path(output))
    
    # Output to a file so we don't need to run this again and again to make new graphs
    file_results = open(info_out_file, 'w')
    
    file_results.write("Numerical analysis data for model 38 at " + model_in)
    file_results.write('\n')
    file_results.write("Number of samples in each set (set generated by model and set from held-out test data): " + str(num_samples))
    file_results.write('\n')
    file_results.write("==================================================================================")
    file_results.write('\n')
    file_results.write('\n')
    
    for v in model_lineal_path_vals:
        file_results.write("model, lineal path value: " + str(v))
        file_results.write('\n')

    file_results.write('\n')

    for v in model_two_point_vals:
        file_results.write("model, two-point probability function value: " + str(v))
        file_results.write('\n')

    file_results.write('\n')

    for v in model_porosity_vals:
        file_results.write("model, porosity (calculated using lineal path mean of max) value: " + str(v))
        file_results.write('\n')    

    file_results.write('\n')

    for v in test_lineal_path_vals:
        file_results.write("held-out test data, lineal path value: " + str(v))
        file_results.write('\n') 

    file_results.write('\n')
    
    for v in test_two_point_vals:
        file_results.write("held-out test data, two-point probability function value: " + str(v))
        file_results.write('\n')    

    file_results.write('\n')
    
    for v in test_porosity_vals:
        file_results.write("held-out test data, porosity (calculated using lineal path mean of max) value: " + str(v))
        file_results.write('\n')     

    file_results.close()

elif req_option == 31:
    print("*** Making graphs from the data made in option 30 (which used a saved model and the held-out test data). ***")
    
    timestamp_wanted = '1652075867'
    info_in_file = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis_of_model_38/" + timestamp_wanted + "_model_38_numerical_data.txt"
    out_directory = "C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/k-s_analysis_of_model_38/"
 
    f = open(info_in_file, 'r')
    
    model_lineal_path_vals = []
    model_two_point_vals = []
    model_porosity_vals = []
    
    test_lineal_path_vals = []
    test_two_point_vals = []
    test_porosity_vals = []   
    
    # Pull data from file into memory
    for linea in f:
        if linea.strip().find("model, lineal path value:") != -1:
            colon_pos = linea.rfind(":")
            val = float(linea[colon_pos+1:])
            model_lineal_path_vals.append(val)

        elif linea.strip().find("model, two-point probability function value:") != -1:
            colon_pos = linea.rfind(":")
            val = float(linea[colon_pos+1:])
            model_two_point_vals.append(val)            

        elif linea.strip().find("model, porosity (calculated using lineal path mean of max) value:") != -1:
            l_bracket_pos = linea.rfind("[")
            val = linea[l_bracket_pos+1:-2]
            model_porosity_vals.append(float(val))            

        elif linea.strip().find("held-out test data, lineal path value:") != -1:
            colon_pos = linea.rfind(":")
            val = float(linea[colon_pos+1:])
            test_lineal_path_vals.append(val)            

        elif linea.strip().find("held-out test data, two-point probability function value:") != -1:
            colon_pos = linea.rfind(":")
            val = float(linea[colon_pos+1:])
            test_two_point_vals.append(val)            
            
        elif linea.strip().find("held-out test data, porosity (calculated using lineal path mean of max) value:") != -1:
            l_bracket_pos = linea.rfind("[")
            val = linea[l_bracket_pos+1:-2]
            test_porosity_vals.append(float(val))            

    # lineal path value
    histfile = out_directory + timestamp_wanted + "_model_38_lineal_path_histogram.png"
    linefile = out_directory + timestamp_wanted + "_model_38_lineal_path_line.png"
    make_graph_from_lists(model_lineal_path_vals, test_lineal_path_vals, histfile, linefile, "Comparison of Mean Lineal Path for Model and Test Data", "Mean Lineal Path", "Samples", "upper left")

    # two-point probability function
    histfile = out_directory + timestamp_wanted + "_model_38_two_pt_prob_histogram.png"
    linefile = out_directory + timestamp_wanted + "_model_38_two_pt_prob_line.png"
    make_graph_from_lists(model_two_point_vals, test_two_point_vals, histfile, linefile, "Comparison of Two-Point Probability for Model and Test Data", "Mean Two-Point Probability", "Samples", "upper left")

    # porosity (calculated using lineal path mean of max)
    histfile = out_directory + timestamp_wanted + "_model_38_porosity_histogram.png"
    linefile = out_directory + timestamp_wanted + "_model_38_porosity_line.png"
    make_graph_from_lists(model_porosity_vals, test_porosity_vals, histfile, linefile, "Comparison of Lineal Path-Derived Porosity for Model and Test Data", "Mean Porosity", "Samples", "upper right")
    
    print("Histograms and line graphs saved to " + out_directory)
    
elif req_option == 32:
    print("Making explanatory images of features")
    
    file_in = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/test_samples/374_01_06_256.mat'
    files_saved_out = 'C:/!data/college/2022 Spring/CISC867 - Soft Materials/goreCode/working/867-team-gore/analysis/figures_out/explain_feature_volumes/'
    
    incoming_data = loadmat(file_in)['bin']

    i = incoming_data[64:128,:64,64:128]
    matz = incoming_data[64:128,:64,64:128]
    
    """
    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')
    ax.voxels(i, facecolors='oldlace', edgecolor='k', linewidth=0.01)

    file_name_voxels = '374_01_06_256_subvolume_material.png'
    file_out  = os.path.join(files_saved_out, file_name_voxels)
    
    plt.savefig(file_out, dpi=300)
    plt.close()
    """

    i = np.invert(matz)
    
    for x in range(len(i)):
        for y in range(len(i)):
            for z in range(len(i)):
                if matz[x,y,z] == 1:
                     i[x,y,z] = 0
                else:
                    i[x,y,z] = 1 
        
    
    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')

    # https://matplotlib.org/stable/gallery/mplot3d/voxels.html   
    i2 = np.empty(i.shape, dtype=object)
    matz2 = np.empty(i.shape, dtype=object)
    for x in range(len(i2)):
        for y in range(len(i2)):
            for z in range(len(i2)):
                if i[x,y,z] == 1:
                     i2[x,y,z] = True
                else:
                    i2[x,y,z] =  False

    for x in range(len(matz2)):
        for y in range(len(matz2)):
            for z in range(len(matz2)):
                if matz[x,y,z] == 1:
                     matz2[x,y,z] = True
                else:
                    matz2[x,y,z] =  False
    
    facecolors = np.where(matz2, 'peachpuff', 'oldlace') 
    edgecolors = np.where(matz2, '#BFAB6E', '#7D84A6')
    filled = np.ones(matz2.shape)
    
    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)
    
    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95
    
    ax = plt.figure().add_subplot(projection='3d')
    plt.grid(False)
    plt.axis('off')
    
    #ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    
    
    """
    voxelarray = i2 | matz2
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[i2] == 'peachpuff'
    colors[matz2] == 'oldlace'
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k', linewidth=0.01)
    """
  
    """  
    print("A")
    ax.voxels(i, facecolors='peachpuff', edgecolor='k', linewidth=0.01)
    print("B")
    ax.voxels(incoming_data, facecolors='oldlace', edgecolor='k', linewidth=0.01)
    print("C")
    """

    ax.voxels(matz2, facecolors='#BFAB6E', edgecolor='k', linewidth=0.01)

    #file_name_voxels = '374_01_06_256_subvolume_void_and_material.png'
    file_name_voxels = '374_01_06_256_subvolume_material.png'
    file_out  = os.path.join(files_saved_out, file_name_voxels)
    
    plt.savefig(file_out, dpi=300)
    plt.close()
    
    


else:
    print("That was not an option.")

print("Done.")

