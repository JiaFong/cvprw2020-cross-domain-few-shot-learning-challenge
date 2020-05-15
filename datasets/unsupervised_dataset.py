import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *

identity = lambda x:x
class SimpleDataset:
    def __init__(self, name, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        uls_file = "./unsupervised-track/Unsupervised_" + name + ".txt"
 
        with open(uls_file, 'r') as f:
            uls_list = f.read().splitlines()

        if name == 'CropDisease':
            path = CropDisease_path + "/dataset/train/"
        elif name == 'EuroSAT':
            path = EuroSAT_path + '/'
        elif name == 'ISIC':
            path = ISIC_path + "/ISIC2018_Task3_Training_Input/"
        else:
            path = ChestX_path + '/images/'

        for idx, name in enumerate(uls_list):
            img_path = path + name
            self.meta['image_names'].append(img_path)

        self.meta['image_labels'] = [-1] * len(self.meta['image_names'])

    def __getitem__(self, i):

        img = self.transform(Image.open(self.meta['image_names'][i]).convert('RGB'))
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)

        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 

        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':

            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, name, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)


    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(self.name, transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

