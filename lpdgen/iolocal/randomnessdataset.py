import torch
import torch.utils.data as torchdata
import pytorch_lightning as lightning
import numpy as np


class randomnessdataset(torchdata.Dataset):
    def __init__(self):
        super(torchdata.Dataset, self).__init__()
        
        
    def __len__(self):
        return 500      # Plausibly long. We have an infinite supply, after all.
    
    def __getitem__(self, index):
        batch_size_loc = 16
        rand_input = torch.randn(64 * 4 * 4 * 4, device=self.device)
        rand_input = rand_input.view(64, 4,4,4)
        return rand_input


        
