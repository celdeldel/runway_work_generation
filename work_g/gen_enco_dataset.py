"""
@author: celdel

file with the class to load the dataset
you should double check that the encodings are related to the right pic
this was the case 


"""

import os
import numpy as np
import torch
import torchvision
import pandas as pd
import PIL
from PIL import Image
from skimage import io, transform
import torchvision.transforms as transforms
from torchvision.utils import save_image


from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset, DataLoader

class FaceEncodsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.encods_frame = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.names = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.encods_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.names[idx])
        image = io.imread(img_name)
        encods = self.encods_frame.iloc[idx, 1:].as_matrix()
        encods = encods.astype('float').reshape(1, -1) 
        sample = {'image': image, 'encods': encods}

        if self.transform:
            sample = self.transform(sample)

        return sample
 
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, encods = sample['image'], sample['encods']        
        img = transform.resize(image, self.output_size)
        return {'image': img, 'encods': encods}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, encods = sample['image'], sample['encods']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'encods': torch.from_numpy(encods)}    

