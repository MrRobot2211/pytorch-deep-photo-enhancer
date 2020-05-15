import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision.datasets import ImageFolder
import albumentations as albu
from albumentations import torch as AT
import pandas as pd
from libs.custom_transforms import PadDifferentlyIfNeeded
from libs.constant import *
from libs.model import *

#import segmentation_models_pytorch as smp

#smp.encoders.get_preprocessing_fn()


class ImageDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__( root, transform=transform)

        # def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
        #             transform = None,
        #             preprocessing=None):
        #     self.df = df
        #     if datatype != 'test':
        #     self.data_folder = f"{path}/train_images"
            
        #self.img_ids = img_ids
        #self.transforms = transforms
        #self.preprocessing = preprocessing
    def  _make_mask(self,img,output_size=(512,512)):
        image_width, image_height = output_size
        crop_height, crop_width = img.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        output = np.zeros(output_size)
        output[crop_left:crop_left + crop_width, crop_top:crop_top +crop_width ] = 1
        return output

    def __getitem__(self, index):
        
        path, target = self.samples[index]
        sample = self.loader(path)
        #mask = self._make_mask( sample)
        sample = np.array(sample)
        mask = np.ones(sample.shape[:-1])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       # augmented = self.transform(image=sample, mask=mask)
        augmented = self.transform(image=sample, mask=mask)
        
        img = augmented['image']
        mask = augmented['mask']

        # if self.preprocessing:
        #     preprocessed = self.preprocessing(image=img, mask=mask)
        #     img = preprocessed['image']
        #     mask = preprocessed['mask']
            
        return img, mask

    # def __len__(self):
    #     return len(self.img_ids)



def data_loader_mask():
    """
    Converting the images for PILImage to tensor,
    so they can be accepted as the input to the network
    :return :
    """
    print("Loading Dataset")
    #transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation='PIL.Image.ANTIALIAS'), transforms.ToTensor()])
    #transform = transforms.Compose([
    # you can add other transformations in this list
   # transforms.CenterCrop(512),
   # transforms.ToTensor()  ])
    default_transform = albu.Compose([ PadDifferentlyIfNeeded(512,512,mask_value=0)
    , AT.ToTensor()])
  
    transform = albu.Compose([ albu.RandomRotate90(1.0)
    , albu.HorizontalFlip(0.5),PadDifferentlyIfNeeded(512,512,mask_value=0), AT.ToTensor()])
  
    testset_gt = ImageDataset(root=TEST_ENHANCED_IMG_DIR , transform=default_transform)
    trainset_1_gt = ImageDataset(root=ENHANCED_IMG_DIR, transform=transform)
    trainset_2_gt = ImageDataset(root=ENHANCED2_IMG_DIR, transform=transform)

    testset_inp = ImageDataset(root=TEST_INPUT_IMG_DIR , transform=default_transform)
    trainset_1_inp = ImageDataset(root=INPUT_IMG_DIR , transform=transform)
    trainset_2_inp = ImageDataset(root=INPUT2_IMG_DIR, transform=transform)

   
    train_loader_cross = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_1_inp,
            trainset_2_gt
        ),num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ConcatDataset(
           
            testset_inp,
            testset_gt
        ),num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=False
    )
    print("Finished loading dataset")

    return  train_loader_cross, test_loader