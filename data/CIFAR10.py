import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

class CIFAR10(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_path = kwargs.get('data_path')
        self.batch_size = kwargs.get('batch_size')
        self.test_batch_size = kwargs.get('test_batch_size')
        self.num_workers = kwargs.get('num_workers')
        self.shuffle = kwargs.get('shuffle')
        self.train_transform = lambda img : A.from_dict(kwargs.get('train_transform'))(image=np.array(img))['image'] 
        self.test_transform = lambda img : A.from_dict(kwargs.get('test_transform'))(image=np.array(img))['image'] 

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, train=True, download=True)
        datasets.CIFAR10(root=self.data_path, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_val_data = datasets.CIFAR10(root=self.data_path, train=True, download=False, transform=self.train_transform)
            self.train_data, self.val_data = random_split(train_val_data, [40000, 10000])
        
        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_data = datasets.CIFAR10(root=self.data_path, train=False, download=False, transform=self.test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch_size, num_workers=self.num_workers)

    # class Dataset(Dataset):
    #     def __init__(self, data_path, transform=None):
    #         self.data_path = data_path
    #         self.transform = transform
            
    #         self.image_files = [f for f in os.listdir(self.data_path) if '_gt' not in f]
            
    #     def __len__(self):
    #         return len(self.image_files)
            
    #     def __getitem__(self, idx):
    #         image_file = self.image_files[idx]
    #         mask_file = self.image_files[idx].replace(".png", "_gt.png")
            
    #         image_path = os.path.join(self.data_path, image_file)
    #         mask_path = os.path.join(self.data_path, mask_file)
            
    #         image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    #         mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
            
    #         image[(mask == 0), :] = 0
            
    #         if self.transform:
    #             augmented = self.transform(image=image, mask=mask)
    #             image = augmented['image']
    #             mask = augmented['mask']
            
    #         return image, mask.long(), idx