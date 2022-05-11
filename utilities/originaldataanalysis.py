import matplotlib.pyplot as plt
import os
import numpy as np
import GooseEYE 
import time
import glob
import math
from hdf5storage import loadmat

class originaldataanalyzer():
    def __init__(self): 
        """
        We'll make as many subvolumes as we can from the main volume of every sample. Our
        requirement is that a subvolume needs at least 2 void (pore) voxels.
        
        pathloc: path to the data; files will be in MAT format.
        """
        
        self.PNGs_wanted = False
        self.alpha_const = 0.05

        # Where to read the generated volumes (the artificial materials)
        self.file_path_input = '../analysis/test_samples/'
        
        # Where to save the analysis 
        self.file_path_analysis = '../analysis/results/' 
        
        
        self.ts_start = time.time()
        self.timestamp1 = str(round(self.ts_start))        # Makes it harder to accidentally overwrite things
        
        print("Timestamp: " + self.timestamp1)
        
        
        ##########################################################################################
        
        self.input_files = glob.glob(os.path.join(self.file_path_input, '*.mat'))
        
        # Where to save the analysis
        self.file_name_analysis = self.timestamp1 + '_analysis_of_held_out.txt'
        self.file_results = open(os.path.join(self.file_path_analysis, self.file_name_analysis), '+w')
        
        self.file_results.write('Mathematical analysis run')
        self.file_results.write('\n')
        self.file_results.write('This is the analysis of the held-out test samples.')
        self.file_results.write('\n')
        self.file_results.write('Timestamp: ' + self.timestamp1)
        self.file_results.write('\n')
        self.file_results.write('\n')
       
        self.chunksize = 64           # Size of the block we want to extract from the full-sized voxel "image"
        self.sizeoforiginal = 256     # Size of the full-sized image

        self.file_results.write('Locations of subvolumes in the files =====================================')
        self.file_results.write('\n')
        
        self.pathloc = self.file_path_input
        
        self.filelist = os.listdir(self.file_path_input)
        
        self.currentmaterial = None         # We'll hold the current rock here, so we don't load it over and over!
        self.currentfileloc = ""            # Keep the file name and location of the current rock here, for comparison.
        
        print(self.filelist)
        
        # Load data into memory
        self.ready_data = []
        
        requiredpore = 2        # how many pore voxels do we require?
        filecount = 0
        
        for index in range(len(self.filelist)):
            # Try to make subvolumes for every file's data
            itm = loadmat(os.path.join(self.pathloc, self.filelist[index]))['bin']
            
            for newleftA in range(itm.shape[0] // self.chunksize):
                for newfrontA in range(itm.shape[0] // self.chunksize):
                    for newtopA in range(itm.shape[0] // self.chunksize):
                        newleft = newleftA * self.chunksize
                        newfront = newfrontA * self.chunksize
                        newtop = newtopA * self.chunksize
                        subset_of_item = itm[newleft:newleft+self.chunksize, newtop:newtop+self.chunksize, newfront:newfront+self.chunksize]
                        
                        numporevoxels = ((subset_of_item.shape[0])**3) - np.sum(subset_of_item)
                        
                        if (numporevoxels > requiredpore):
                            # Store the file name in location for later use
                            self.ready_data.append([os.path.join(self.pathloc, self.filelist[index]), newleft, newtop, newfront])
                            print(str(filecount) + ", " + str([os.path.join(self.pathloc, self.filelist[index]), newleft, newtop, newfront]))
                            
                            self.file_results.write(str(filecount) + ", " + str([os.path.join(self.pathloc, self.filelist[index]), newleft, newtop, newfront]))
                            self.file_results.write('\n')
                            
                            filecount += 1

        self.file_results.write('\n')
        
        self.lineal_path()
        self.two_pt_prob()
        self.lineal_path_mean_all()
        self.two_pt_prob_mean_all()


    def get_one_subvolume(self, index):
    
        currfilelocation =  self.ready_data[index][0]
        
        if currfilelocation != self.currentfileloc:
            # We're on to a new file, so load it.
            self.currentmaterial = loadmat(currfilelocation)['bin']
            self.currentfileloc = currfilelocation
        
        # Get the chunk of the file
        newleft = self.ready_data[index][1]
        newtop = self.ready_data[index][2]
        newfront = self.ready_data[index][3]
        subset_of_item = self.currentmaterial[newleft:newleft+self.chunksize, newtop:newtop+self.chunksize, newfront:newfront+self.chunksize]
        
        subset_of_item_side = subset_of_item.shape[0]
        subset_of_item_reshape = np.reshape(subset_of_item, (1, subset_of_item_side, subset_of_item_side, subset_of_item_side))
        subset_of_item_reshape = subset_of_item_reshape.astype(np.float32)
        
        return (self.currentfileloc, index, subset_of_item_reshape)
    
    
    def lineal_path(self):
        # Lineal path function setup
        lp_sum = 0
        
        self.file_results.write("Lineal path function - porosity (mean of max) ===========================================")
        self.file_results.write('\n')
        
        print()
        print()
        print("Lineal path function - porosity (mean of max) ===========================================")
        print()
        
        for i in range(len(self.ready_data)):
            curr_vol = self.get_one_subvolume(i)
            incoming_file = curr_vol[2]
            curr_idx = curr_vol[1]
            curr_fileloc = curr_vol[0]
            
            # Force to exactly zero and exactly one so we have a binary volume (material and void).
            massaged_output = np.squeeze(incoming_file)
            massaged_output = np.around(massaged_output)
            
            ######################## Calculate lineal path function for this held-out data volume.
            volume_size = 64        # Volume is 64 voxels across.
            line_length = math.trunc(volume_size / 2) - 1
            
            lp_sum = 0
            for i in range(volume_size):
                lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
                lp_sum += max(lpf)
        
            # Write out the lineal path function result for this generated volume.    
            lp_mean = lp_sum/volume_size
            self.file_results.write(curr_fileloc + ", " + str(curr_idx) + ", " + str(1 - lp_mean[0]))
            self.file_results.write('\n')
        
            print(curr_fileloc + ", " + str(curr_idx) + ", " + str(1 - lp_mean[0]))

        self.file_results.write("================================================================")
        self.file_results.write('\n')


    def two_pt_prob(self):
        self.file_results.write("")
        self.file_results.write('\n')
        self.file_results.write("2-point probability function - porosity (mean of max) ===========================================")
        self.file_results.write('\n')
        
        print()
        print()
        print("2-point probability function - porosity (mean of max) ===========================================")
        print()
        
        for i in range(len(self.ready_data)):
            curr_vol = self.get_one_subvolume(i)
            incoming_file = curr_vol[2]
            curr_idx = curr_vol[1]
            curr_fileloc = curr_vol[0]
            
            # Force to exactly zero and exactly one so we have a binary volume (material and void).
            massaged_output = np.squeeze(incoming_file)
            massaged_output = np.around(massaged_output)
            
            ######################## Calculate 2-point probability function for this held-out volume.
            volume_size = 64        # Volume is 64 voxels across.
            line_length = math.trunc(volume_size / 2) - 1
            
            sum = 0
            for i in range(volume_size):  
                fn_goose = GooseEYE.S2((line_length, 1), massaged_output[i], massaged_output[i]) #31 is the distance over two points(white and black), (1, 31) gives us (x, y) different axis
                sum += max(fn_goose)
                
            mean = sum/volume_size
            #find the mean of max value: the mean (1 - porosity) in the third dimension
            
            self.file_results.write(curr_fileloc + ", " + str(curr_idx) + ", " + str(1 - mean[0]))
            self.file_results.write('\n')
            
            print(curr_fileloc + ", " + str(curr_idx) + ", " + str(1 - mean[0]))
            
        self.file_results.write("================================================================")
        self.file_results.write('\n')
        ######################## Done calculating the 2-point probability function.

    def lineal_path_mean_all(self):
        # Lineal path function setup
        lp_sum = 0
        
        self.file_results.write('\n')
        self.file_results.write("Lineal path function - mean of all ===========================================")
        self.file_results.write('\n')
        
        print()
        print()
        print("Lineal path function - mean of all ===========================================")
        print()
        
        for i in range(len(self.ready_data)):
            curr_vol = self.get_one_subvolume(i)
            incoming_file = curr_vol[2]
            curr_idx = curr_vol[1]
            curr_fileloc = curr_vol[0]
            
            # Force to exactly zero and exactly one so we have a binary volume (material and void).
            massaged_output = np.squeeze(incoming_file)
            massaged_output = np.around(massaged_output)
            
            ######################## Calculate lineal path function for this held-out data volume.
            volume_size = 64        # Volume is 64 voxels across.
            line_length = math.trunc(volume_size / 2) - 1
            
            lp_sum = 0
            fn_len = 0
            for i in range(volume_size):
                lpf = GooseEYE.L((line_length, 1), massaged_output[i])  # 31 is the length of the line, (1, 31) gives us (x, y) different axis
                lp_sum += np.sum(lpf)
                fn_len = len(lpf)
        
            # Write out the lineal path function result for this generated volume.    
            lp_mean = (lp_sum/fn_len)/volume_size
            self.file_results.write(curr_fileloc + ", " + str(curr_idx) + ", " + str(lp_mean))
            self.file_results.write('\n')
        
            print(curr_fileloc + ", " + str(curr_idx) + ", " + str(lp_mean))

        self.file_results.write("================================================================")
        self.file_results.write('\n')

    def two_pt_prob_mean_all(self):
        self.file_results.write("")
        self.file_results.write('\n')
        self.file_results.write("2-point probability function - mean of all ===========================================")
        self.file_results.write('\n')
        
        print()
        print()
        print("2-point probability function - mean of all ===========================================")
        print()
        
        for i in range(len(self.ready_data)):
            curr_vol = self.get_one_subvolume(i)
            incoming_file = curr_vol[2]
            curr_idx = curr_vol[1]
            curr_fileloc = curr_vol[0]
            
            # Force to exactly zero and exactly one so we have a binary volume (material and void).
            massaged_output = np.squeeze(incoming_file)
            massaged_output = np.around(massaged_output)
            
            ######################## Calculate 2-point probability function for this held-out volume.
            volume_size = 64        # Volume is 64 voxels across.
            line_length = math.trunc(volume_size / 2) - 1
            
            sum = 0
            fn_len = 0            
            for i in range(volume_size):  
                fn_goose = GooseEYE.S2((line_length, 1), massaged_output[i], massaged_output[i]) #31 is the distance over two points(white and black), (1, 31) gives us (x, y) different axis
                sum += np.sum(fn_goose)
                fn_len = len(fn_goose)
                
            mean = (sum/fn_len)/volume_size
            #find the mean all values
            
            self.file_results.write(curr_fileloc + ", " + str(curr_idx) + ", " + str(mean))
            self.file_results.write('\n')
            
            print(curr_fileloc + ", " + str(curr_idx) + ", " + str(mean))
            
        self.file_results.write("================================================================")
        self.file_results.write('\n')
        ######################## Done calculating the 2-point probability function.        



def main():
    ###  Generate several subvolumes for the held-out data, then run analysis against it #######################################################################################
    print("Original-data analysis starting - mathematical analysis.")
    print("This is the analysis of the held-out test samples.")

    local_start = time.time()
    
    worksite = originaldataanalyzer()


    total_time = time.time() - local_start
    
    worksite.file_results.write('\n')
    worksite.file_results.write('==================================')
    worksite.file_results.write('\n')
    worksite.file_results.write("Wall time expended: " + str(total_time) + " seconds.")
    worksite.file_results.write('\n')
    worksite.file_results.close()
    
    print("Model analysis done.")
    print("Wall time expended: " + str(total_time) + " seconds.")
    
    ### END: Generate an output for a saved model #######################################################################################

if __name__ == "__main__":
    main()
    