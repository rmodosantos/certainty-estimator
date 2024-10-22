import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from imblearn.over_sampling import RandomOverSampler
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
        #image = image.repeat(3,1,1)
        
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


# Generate testing datasets

# Create random image indices given fractions of data
def create_indices(sets_fractions):
    dataset_path = "/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/MRI_dataset_2/"

    #load file with labels
    with h5.File(os.path.join(dataset_path,'labels.h5'),'r') as label_file:
        labels = torch.tensor(label_file['label'])

    #load indices of small images
    inds_small = np.load(os.path.join(dataset_path,'small_images_inds.npy'))
    inds_all = np.setdiff1d(np.arange(1,labels.size()[0]),np.where(inds_small)[0])
    
    with h5.File(os.path.join(dataset_path,'labels.h5'),'r') as label_file:
        labels = torch.tensor(label_file['label'],dtype=torch.long)-1
    
    
    ros = RandomOverSampler(random_state=42)
    
    inds_datasets = random_split(inds_all,sets_fractions,generator=torch.Generator().manual_seed(42))
    
    #inds_res = [1,1]
    inds_res = []
    for i in range(len(sets_fractions)):
        labels_set = labels[inds_datasets[i]]
        inds_tm,labels_res = ros.fit_resample(torch.reshape(torch.tensor(inds_datasets[i]),(-1,1)),labels_set)
        inds_res.append(inds_tm)
        inds_res[i] = torch.tensor(inds_res[i])
        inds_res[i] = inds_res[i][torch.randperm(inds_res[i].size(0),generator=torch.Generator().manual_seed(12))]

    return inds_res


def make_masked_testset(fraction,reduction,dataset_nr,dev):
    """Instantiates a dataset masking specified fraction of pixels"""
    
    inds_res = create_indices([0.6,0.22,0.18])
    
    dataset_path = "/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/MRI_dataset_2/"

    # Define a custom function to mask the image
    def random_zero_mask(image, dev, fraction):
        """
        Create a random mask where a fraction of pixels are set to zero.

        Args:
            image (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width).
            fraction (float): Fraction of pixels to set to zero.

        Returns:
            torch.Tensor: Random mask tensor with the same shape as the input image.
        """
        
        mask = torch.rand((int(512/reduction),int(512/reduction)),device=dev) < fraction
        mask = torch.where(mask,0,1)
        
        #define resizing transform
        resize_transform = transforms.Resize((512,512),interpolation=transforms.InterpolationMode.NEAREST_EXACT,antialias=False)
        
        mask = resize_transform(mask.unsqueeze(0))
        
        masked_image = image * mask
        
        return masked_image

    def mask_transform(fraction):

        mask = transforms.Lambda(lambda x: random_zero_mask(x, dev, fraction))
        return mask
  
    add_mask_transform = mask_transform(fraction)

    translation=(0.1, 0.1)
    scales=(0.9, 1.05)
    degrees=(-10, 10)

    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0,translate=translation),
    transforms.RandomAffine(degrees=0,scale=scales),
    transforms.RandomAffine(degrees=degrees),
    add_mask_transform
    ])

    return MRI_dataset(dataset_path, inds_res[dataset_nr], dev,transforms=transform)


def make_noise_testset(noise_std,dataset_nr,dev):
    """Instantiates a dataset masking specified fraction of pixels"""
    
    dataset_path = "/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/MRI_dataset_2/"
    inds_res = create_indices([0.6, 0.22, 0.18])

   # Define a custom function to add noise to the image
    def add_noise(image, dev, mean=0, std=0.1):
        """
        Add Gaussian noise to the input image.
        """

        #image = image.cpu().numpy()

        #std = torch.randint(std[0], std[1],(1,))
        # Add Gaussian noise
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise

#         # Convert numpy array back to tensor
#         noisy_image = torch.tensor(noisy_image,dtype=torch.float).to(dev)

        return noisy_image

    def noise_transform(std):

        add_noise_transform = transforms.Lambda(lambda x: add_noise(x, dev, std=std))
        return add_noise_transform

    add_noise_transform = noise_transform(noise_std)

    #transform = transforms.Compose([add_noise_transform])
    
    translation=(0.1, 0.1)
    scales=(0.9, 1.05)
    degrees=(-10, 10)
    
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0,translate=translation),
    transforms.RandomAffine(degrees=0,scale=scales),
    transforms.RandomAffine(degrees=degrees),
    add_noise_transform
    ])
    
    return MRI_dataset(dataset_path, inds_res[dataset_nr], dev,transforms=transform)