import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Subset

class CIFAR10(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs.get('data_path')
        self.batch_size = kwargs.get('batch_size')
        self.test_batch_size = kwargs.get('test_batch_size')
        self.num_workers = kwargs.get('num_workers')
        self.shuffle = kwargs.get('shuffle')
        self.k_folds = kwargs.get('k_folds')
        self.k = kwargs.get('k')
        self.train_transform = lambda img : A.from_dict(kwargs.get('train_transform'))(image=np.array(img))['image'] 
        self.test_transform = lambda img : A.from_dict(kwargs.get('test_transform'))(image=np.array(img))['image'] 

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, train=True, download=True)
        datasets.CIFAR10(root=self.data_path, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.k < 0:
                train_val_data = datasets.CIFAR10(root=self.data_path, train=True, download=False, transform=self.train_transform)
                self.train_data, self.val_data = random_split(train_val_data, [40000, 10000])
            else:
                train_val_data = datasets.CIFAR10(root=self.data_path, train=True, download=False, transform=self.train_transform)
                
                # Calculate the size of each fold
                fold_size = len(train_val_data) // self.k_folds
                val_start = fold_size * self.k
                val_end = val_start + fold_size
                
                # Split the data into train and validation sets
                indices = list(range(len(train_val_data)))
                train_indices = indices[:val_start] + indices[val_end:]
                val_indices = indices[val_start:val_end]
                
                self.train_data = Subset(train_val_data, train_indices)
                self.val_data = Subset(train_val_data, val_indices)
        
        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_data = datasets.CIFAR10(root=self.data_path, train=False, download=False, transform=self.test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch_size, num_workers=self.num_workers)