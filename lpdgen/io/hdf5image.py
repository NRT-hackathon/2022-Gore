"""
hdf5image.py

Module to handle reading and writing of image data in the HDF5 format
"""

import numpy
import h5py
import torch


def is_hdf5(filename):
    """Check if a file is a hdf5 file
    
    Parameters
    ----------
    filename: str or pathlib.Path
        File name including extension of input file

    extension: List(str)
        List of possible extensions for file type
    """
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])


def load_hdf5(filepath, hdf_key='data'):
    """Load a HDF5 image
    
    Parameters
    ----------
    filepath: str or pathlib.Path
        Image file path 

    hdf_key: str
        The key within the HDF hierarchy holding the image data.
        Default value is `data`

    Returns
    -------
    torch_img : torch.Tensor
        Image in torch tensor format
    """
    try:
        img = None
        with h5py.File(filepath, "r") as f:
            img = f[hdf_key][()]
        img = numpy.expand_dims(img, axis=0)
        torch_img = torch.Tensor(img)
        torch_img = torch_img.div(255).sub(0.5).div(0.5)
        return torch_img
    except Exception:
        raise


def save_hdf5(tensor, filename, hdf_key='data'):
    """Save a Tensor into a HDF5 image file.

    If given a mini-batch tensor, this will save the tensor as a grid of 
    images.

    Parameters
    ----------
    tensor: torch.Tensor
        Image data

    filename: str or pathlib.Path
        File name including full path of where the image will be saved

    hdf_key: str
        The key within the HDF hierarchy holding the image data.
        Default value is `data`
    """
    tensor = tensor.cpu()
    ndarr = tensor.mul(0.5).add(0.5).mul(255).byte().numpy()
    with h5py.File(filename, 'w') as f:
        f.create_dataset(hdf_key, data=ndarr, dtype="i8", compression="gzip")