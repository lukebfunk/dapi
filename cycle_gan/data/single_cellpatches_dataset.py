from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
import pandas as pd
# from PIL import Image
import torch
from torchvision import transforms
import zarr


class SingleCellpatchesDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_zarr = zarr.open('/nrs/funke/funkl/data/dapi/nontargeting_test_samples.zarr','r')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_img = self.A_zarr[index,:,64:-64,64:-64] # only need central 128x128 crop
        assert A_img.dtype == np.uint16
        A = torch.from_numpy((A_img/(2**16 - 1)).astype(np.float32))
        return {'A': A, 'A_paths': index} # index is the relevant metadata, all the same zarr path

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_zarr.shape[0]
