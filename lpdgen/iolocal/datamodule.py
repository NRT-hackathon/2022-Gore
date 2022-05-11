import os
import pathlib
import torch.utils.data as torchdata
import pytorch_lightning as lightning

from typing import Any, Callable, List, Optional, Union
from .hdf5image import load_hdf5, is_hdf5

from .materialdataset import *


class DRPDataset(torchdata.Dataset):
    def __init__(
        self, 
        data_dir: Union[str, pathlib.Path], 
        input_transform: Optional[Any] = None, 
        target_transform: Optional[Any] = None
    ):
        super(DRPDataset, self).__init__()
        self.image_filenames = [ os.join(data_dir, x) 
                                 for x in os.listdir(data_dir) if is_hdf5(x) ]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_hdf5(self.image_filenames[index])
        target = None
        return input

    def __len__(self):
        return len(self.image_filenames)



class DRPDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, pathlib.Path] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 4,
        normalize: bool = False,
        batch_size: int = 16,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super(DRPDataModule, self).__init__(*args, **kwargs)
        self.data_dir = data_dir 
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last


    def setup(self, stage: Optional[str] = None):
        """Data operations to be performed on every GPU"""
        pass

    def train_dataloader(self):
        """Generate the training Dataloader object"""
        return torchdata.DataLoader(materialdataset('./data/train'), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Generate the validation Dataloader object"""
        return torchdata.DataLoader(materialdataset('./data/val'), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        """Generate the test Dataloader object"""
        return torchdata.DataLoader(materialdataset('./data/test'), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def teardown(self, stage: Optional[str] = None):
        """Clean-up when the run is finished"""
        pass

    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""
        pass
