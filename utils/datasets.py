import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from imblearn.over_sampling import RandomOverSampler
import os
import h5py as h5
import numpy as np

# Dataset classes used in the project

class MRI_dataset_random(Dataset):
    """Dataset class for random MRI data sampling using balanced data retrieval."""

    def __init__(self, data_folder, inds, dev, transforms=None):
        """Initialize the MRI dataset.

        Args:
            data_folder (str): Path to the data folder
            inds (tensor): Indices for data selection
            dev (str): Device for tensor operations
            transforms (callable, optional): Optional transforms to apply
        """
        
        self.data_folder = data_folder
        with h5.File(os.path.join(data_folder,'labels.h5'),'r') as label_file:
            labels = torch.tensor(label_file['label'],dtype=torch.long)-1
            
        self.labels = labels[inds].to(dev)
        self.inds = inds
        self.transforms = transforms
        self.dev = dev
        
    def __len__(self):
        return sum(self.labels==1) * 3
    
    def __getitem__(self,idx):
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item

        Returns:
            tuple: (image, label, idx)
        """
        
        labl = []
        for i in torch.arange(0,3):
            labl.append(torch.where(self.labels==i)[0])
            
        ind = torch.concatenate((labl[0][torch.randint(0,len(labl[0]),(1,))],
                                 labl[1][torch.randint(0,len(labl[1]),(1,))],
                                 labl[2][torch.randint(0,len(labl[2]),(1,))]))
        idx_labels = int(ind[torch.randint(0,3,(1,))].cpu())
        idx = int(self.inds[idx_labels])
        
        # Get label
        label = self.labels[idx_labels]
        
        # Load image
        file_name = f"{idx+1}.mat"
        image_path = os.path.join(self.data_folder,file_name)
        
        with h5.File(image_path,'r') as imfile:
            image = torch.tensor(np.array(imfile['cjdata']['image']),dtype=torch.float).unsqueeze(0).to(self.dev)
        
        if self.transforms:
            #print(image.size())
            image = self.transforms(image)
        
        return image, label, idx
    
    
    
class MRI_dataset(Dataset):
    """Dataset class for MRI data sampling from given sample indices"""
    
    def __init__(self, data_folder, inds, dev, transforms=None):
        """Initialize the MRI dataset.

        Args:
            data_folder (str): Path to the data folder
            inds (tensor): Indices for data selection
            dev (str): Device for tensor operations
            transforms (callable, optional): Optional transforms to apply
        """
        
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
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item

        Returns:
            tuple: (image, label)
        """
        
        ind_image = int(self.inds[idx])
        
        # Get label
        label = self.labels[idx]
        
        # Load image
        file_name = f"{ind_image+1}.mat"
        image_path = os.path.join(self.data_folder,file_name)
        
        with h5.File(image_path,'r') as imfile:
            image = torch.tensor(np.array(imfile['cjdata']['image']),dtype=torch.float).unsqueeze(0).to(self.dev)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label
    
    



# Generate testing datasets

def create_indices(sets_fractions):
    """Create random image indices given fractions of data.

    Args:
        sets_fractions (list): List of fractions for dataset splitting

    Returns:
        list: List of tensor indices for each dataset split
    """
    
    dataset_path = "/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/MRI_dataset_2/"

    # Load file with labels
    with h5.File(os.path.join(dataset_path,'labels.h5'),'r') as label_file:
        labels = torch.tensor(label_file['label'])

    # Load indices of small images to exclude
    inds_small = np.load(os.path.join(dataset_path,'small_images_inds.npy'))
    inds_all = np.setdiff1d(np.arange(1,labels.size()[0]),np.where(inds_small)[0])
    
    # Reload labels and adjust them to 0-based indexing
    with h5.File(os.path.join(dataset_path,'labels.h5'),'r') as label_file:
        labels = torch.tensor(label_file['label'],dtype=torch.long)-1
    
    # Initialize random oversampler for class balancing
    ros = RandomOverSampler(random_state=42)
    
    # Split indices according to specified fractions
    inds_datasets = random_split(inds_all,sets_fractions,generator=torch.Generator().manual_seed(42))
    
    #inds_res = [1,1]
    inds_res = []
    for i in range(len(sets_fractions)):
        # Get labels for current split
        labels_set = labels[inds_datasets[i]]
        
        # Apply random oversampling to balance classes
        inds_tm,labels_res = ros.fit_resample(torch.reshape(torch.tensor(inds_datasets[i]),(-1,1)),labels_set)
        
        # Convert to tensor and shuffle
        inds_res.append(inds_tm)
        inds_res[i] = torch.tensor(inds_res[i])
        inds_res[i] = inds_res[i][torch.randperm(inds_res[i].size(0),generator=torch.Generator().manual_seed(12))]

    return inds_res


def make_masked_testset(fraction,reduction,dataset_nr,dev):
    """Instantiate a dataset masking specified fraction of pixels.

    Args:
        fraction (float): Fraction of pixels to mask
        reduction (int): Reduction factor for mask template
        dataset_nr (int): Dataset number
        dev (str): Device for tensor operations

    Returns:
        MRIDataset: Dataset with masked images
    """
    
    # Split data into train, validation, and test sets
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
        
        # Create initial random mask template at reduced resolution
        mask = torch.rand((int(512/reduction),int(512/reduction)),device=dev) < fraction
        mask = torch.where(mask,0,1)
        
        # Define resizing transform to original resolution
        resize_transform = transforms.Resize((512,512),interpolation=transforms.InterpolationMode.NEAREST_EXACT,antialias=False)
        
        # Resize mask and apply it to image
        mask = resize_transform(mask.unsqueeze(0))
        masked_image = image * mask
        
        return masked_image

    def mask_transform(fraction):
        """Create a transform function for masking."""
        mask = transforms.Lambda(lambda x: random_zero_mask(x, dev, fraction))
        return mask
  
    # Create masking transform
    add_mask_transform = mask_transform(fraction)
    
    # Define augmentation parameters for further transformations
    translation=(0.1, 0.1)
    scales=(0.9, 1.05)
    degrees=(-10, 10)

    # Combine all transforms
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
    """Instantiate a dataset with added noise.

    Args:
        noise_std (float): Standard deviation of noise
        dataset_nr (int): Dataset number
        dev (str): Device for tensor operations

    Returns:
        MRIDataset: Dataset with noisy images
    """
    
    dataset_path = "/dss/dssfs02/lwp-dss-0001/pr84qa/pr84qa-dss-0000/ricardo/data/MRI_dataset_2/"
    inds_res = create_indices([0.6, 0.22, 0.18])

    def add_noise(image, dev, mean=0, std=0.1):
        """Add Gaussian noise to the input image."""

        # Add Gaussian noise
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise

        return noisy_image

    def noise_transform(std):
        """Create a transform function for adding noise."""
        add_noise_transform = transforms.Lambda(lambda x: add_noise(x, dev, std=std))
        return add_noise_transform
    
    # Create noise transform
    add_noise_transform = noise_transform(noise_std)

    # Define augmentation parameters for further transformations
    translation=(0.1, 0.1)
    scales=(0.9, 1.05)
    degrees=(-10, 10)
    
    # Combine all transforms
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0,translate=translation),
    transforms.RandomAffine(degrees=0,scale=scales),
    transforms.RandomAffine(degrees=degrees),
    add_noise_transform
    ])
    
    return MRI_dataset(dataset_path, inds_res[dataset_nr], dev,transforms=transform)