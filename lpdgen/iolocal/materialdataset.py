import os
import pathlib
import torch
import torch.utils.data as torchdata
import pytorch_lightning as lightning

from typing import Any, Callable, List, Optional, Union
from .hdf5image import load_hdf5, is_hdf5
from pandas._libs import index

from hdf5storage import loadmat
import numpy as np
import random

#import time


class materialdataset(torchdata.Dataset):
    def __init__(self, pathloc): 
        """
        We'll make as many subvolumes as we can from the main volume of every sample. Our
        requirement is that a subvolume needs at least 2 void (pore) voxels.
        
        pathloc: path to the data; files will be in MAT format.
        """
        super(torchdata.Dataset, self).__init__()
        
        self.chunksize = 64           # Size of the block we want to extract from the full-sized voxel "image"
        self.sizeoforiginal = 256     # Size of the full-sized image
        
        self.pathloc = pathloc
        
        self.filelist = os.listdir(self.pathloc)
        
        self.currentmaterial = None         # We'll hold the current rock here, so we don't load it over and over!
        self.currentfileloc = ""            # Keep the file name and location of the current rock here, for comparison.
        
        print(self.filelist)
        
        # Load data into memory
        self.ready_data = []
        
        requiredpore = 2        # how many pore voxels do we require?
        
        for index in range(len(self.filelist)):
            # Try to make subvolumes for every file's data

            print("index: " + str(index))
            
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
                            # Store the file name an location for later use
                            self.ready_data.append([os.path.join(self.pathloc, self.filelist[index]), newleft, newtop, newfront])
                            print([os.path.join(self.pathloc, self.filelist[index]), newleft, newtop, newfront])
        # Done loading data into memory
        
    def __len__(self):
        return len(self.ready_data)
    
    def __getitem__(self, index):

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
        
        return (subset_of_item_reshape)
