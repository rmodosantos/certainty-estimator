import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import h5py as h5
import numpy as np


class MRI_dataset_random(Dataset):
    def __init__(self, data_folder, inds, dev, transforms=None):
        self.data_folder = data_folder
        with h5.File(os.path.join(data_folder,'labels.h5'),'r') as label_file:
            labels = torch.tensor(label_file['label'],dtype=torch.long)-1
            #print(labels[1])
        self.labels = labels[inds].to(dev)
        self.inds = inds
        self.transforms = transforms
        self.dev = dev
        
    def __len__(self):
        return sum(self.labels==1)*3
    
    def __getitem__(self,idx):
        #print(type(self.labels))
        labl = []
        for i in torch.arange(0,3):
            labl.append(torch.where(self.labels==i)[0])
            
        ind = torch.concatenate((labl[0][torch.randint(0,len(labl[0]),(1,))],
                                 labl[1][torch.randint(0,len(labl[1]),(1,))],
                                 labl[2][torch.randint(0,len(labl[2]),(1,))]))
        idx_labels = int(ind[torch.randint(0,3,(1,))].cpu())
        idx = int(self.inds[idx_labels])
        
        #get label
        label = self.labels[idx_labels]
        
        #load image
        file_name = f"{idx+1}.mat"
        image_path = os.path.join(self.data_folder,file_name)
        
        with h5.File(image_path,'r') as imfile:
            image = torch.tensor(np.array(imfile['cjdata']['image']),dtype=torch.float).unsqueeze(0).to(self.dev)
        
        if self.transforms:
            #print(image.size())
            image = self.transforms(image)
            #print(image.size())
            #image = image.squeeze(3)
            #print(image.size())
        #image = image.to(self.dev)
        image = image.repeat(3,1,1)
        
        return image, label, idx
    
    
    
class MRI_dataset(Dataset):
    def __init__(self, data_folder, inds, dev, transforms=None):
        self.data_folder = data_folder
        with h5.File(os.path.join(data_folder,'labels.h5'),'r') as label_file:
            labels = torch.tensor(label_file['label'],dtype=torch.long)-1
            #print(labels[1])
        self.labels = labels[inds].to(dev)
        self.inds = inds
        self.transforms = transforms
        self.dev = dev
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        #print(type(self.labels))
        
        ind_image = int(self.inds[idx])
        
        #get label
        label = self.labels[idx]
        
        #load image
        file_name = f"{ind_image+1}.mat"
        image_path = os.path.join(self.data_folder,file_name)
        
        with h5.File(image_path,'r') as imfile:
            image = torch.tensor(np.array(imfile['cjdata']['image']),dtype=torch.float).unsqueeze(0).to(self.dev)
        
        if self.transforms:
            #print(image.size())
            image = self.transforms(image)
            #print(image.size())
            #image = image.squeeze(3)
            #print(image.size())
        #image = image.to(self.dev)
        #image = image.repeat(3,1,1)
        
        return image, label
    
    
class MRI_dataset_masked(Dataset):
    def __init__(self, data_folder, inds, dev, transforms=None):
        self.data_folder = data_folder
        with h5.File(os.path.join(data_folder,'labels.h5'),'r') as label_file:
            labels = torch.tensor(label_file['label'],dtype=torch.long)-1
            #print(labels[1])
        self.labels = labels[inds].to(dev)
        self.inds = inds
        self.transforms = transforms
        self.dev = dev
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        #print(type(self.labels))
        
        ind_image = int(self.inds[idx])
        
        #get label
        label = self.labels[idx]
        
        #load image
        file_name = f"{ind_image+1}.mat"
        image_path = os.path.join(self.data_folder,file_name)
        
        with h5.File(image_path,'r') as imfile:
            image = torch.tensor(np.array(imfile['cjdata']['image']),dtype=torch.float).unsqueeze(0).to(self.dev)
        
        if self.transforms:
            #print(image.size())
            image = self.transforms(image)
            #print(image.size())
            #image = image.squeeze(3)
            #print(image.size())
        #image = image.to(self.dev)
        #image = image.repeat(3,1,1)
        
        def random_zero_mask(image, dev, max_fraction):
            """
            Create a random mask where a fraction of pixels are set to zero.

            Args:
                image (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width).
                fraction (float): Fraction of pixels to set to zero.

            Returns:
                torch.Tensor: Random mask tensor with the same shape as the input image.
            """
            
            fraction = torch.rand((1,),device=dev)*max_fraction
            
            reduction = 32
            mask = torch.rand((int(512/reduction),int(512/reduction)),device=dev) < fraction
            mask = torch.where(mask,0,1)

            #define resizing transform
            resize_transform = transforms.Resize((512,512),interpolation=transforms.InterpolationMode.NEAREST_EXACT,antialias=False)

            mask = resize_transform(mask.unsqueeze(0))
            masked_image = image * mask

            return masked_image
        
        image = random_zero_mask(image,self.dev,0.95)
        
        return image, label