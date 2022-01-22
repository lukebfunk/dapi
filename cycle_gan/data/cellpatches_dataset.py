"""Dataset class cellpatches

You can specify '--dataset_mode cellpatches' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import numpy as np
import pandas as pd
import random
import torch
from torchvision import transforms
import zarr
# from data.image_folder import make_dataset
# from PIL import Image


class CellPatchesDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--samples_csv', type=str,default='~/gemelli/dataset_info/1_train_samples.csv')
        parser.add_argument('--gene',type=str,required=True)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        
        restrict_query=f'gene_symbol==["nontargeting","{opt.gene}"]'
        metadata = pd.concat([pd.read_csv(f'~/gemelli/dataset_info/1_{d}_samples.csv').query(restrict_query) for d in ['train','test','val']])

        assert metadata['gene_symbol'].nunique()==2
        assert (metadata['gene_symbol']=='nontargeting').sum()>0

        metadata_A = metadata.query('gene_symbol=="nontargeting"')
        metadata_B = metadata.query('gene_symbol!="nontargeting"')
        self.A_paths = metadata_A['store'].tolist()  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        self.B_paths = metadata_B['store'].tolist()
        self.A_arrays = metadata_A['array'].tolist()
        self.B_arrays = metadata_B['array'].tolist()
        self.A_array_indices = metadata_A['array_index'].values.astype(np.int64)
        self.B_array_indices = metadata_B['array_index'].values.astype(np.int64)
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        self.transform = transforms.Compose([
            transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(128)
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        index_A = index % self.A_size
        path_A = '/nrs/funke/funkl/data/'+self.A_paths[index]
        array_A = self.A_arrays[index]
        array_index_A  = self.A_array_indices[index]

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        path_B = '/nrs/funke/funkl/data/'+self.B_paths[index_B]
        array_B = self.B_arrays[index_B]
        array_index_B  = self.B_array_indices[index_B]

        # print(path_A)
        z_A = zarr.open(path_A)
        A_img = z_A[array_A][array_index_A]
        assert A_img.dtype == np.uint16
        A_img = torch.from_numpy((A_img/(2**16 - 1)).astype(np.float32))

        z_B = zarr.open(path_B)
        B_img = z_B[array_B][array_index_B]
        assert B_img.dtype == np.uint16
        B_img = torch.from_numpy((B_img/(2**16 - 1)).astype(np.float32))

        # apply image transformation
        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B, 'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
