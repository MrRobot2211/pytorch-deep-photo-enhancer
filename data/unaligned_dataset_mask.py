import numpy as np
import cv2
import torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
import albumentations as albu
from albumentations import torch as AT
import pandas as pd
from libs.custom_transforms import PadDifferentlyIfNeeded


#import segmentation_models_pytorch as smp

#smp.encoders.get_preprocessing_fn()


class ImageDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__( root, transform=transform)

       

    def __getitem__(self, index):
        
        path, target = self.samples[index]
        sample = self.loader(path)

        sample = np.array(sample)
        mask = np.ones(sample.shape[:-1])

        augmented = self.transform(image=sample, mask=mask)
        
        img = augmented['image']
        mask = augmented['mask']

        return img, mask

    # def __len__(self):
    #     return len(self.img_ids)



