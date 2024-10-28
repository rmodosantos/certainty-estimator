# Datasheet

## Motivation
The dataset was created with the main goal of contributing to the development and testing of new machine learning algorithms for MRI image classification. The choice of this dataset in the context of the current project was motivated by the previously demonstrated high accuracy of commonly used convolutional neural networks, such as Resnets, in tumor classification tasks using this dataset [1].

## Composition
The dataset is composed of 3064 T1-weighted contrast-inhanced images from 233 patients with three kinds of brain tumor: meningioma (708 slices), glioma (1426 slices), and pituitary tumor (930 slices). The dataset is publicly available at: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=7953679

## Collection process
Data was acquired by full MRI scans on patients that were diagnosed with brain tumors. The dataset is a sample of a larger set, not publicly available, to the best of my knowledge.

## Preprocessing/cleaning/labelling
The dataset is split into 4 subsets, and achived in 4 .zip files, each containing 766 slices. The raw data is in Matlab format and is organized as a structure containing raw data, processed data and metadata, as follows:

- cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
- cjdata.PID: patient ID
- cjdata.image: image data
- cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.

## Usage
This datased is intended to be used for developing and testing new machine learning algorithms for MRI image classification.

## Distribution
No copyright restrictions. Please check the dataset link for further details or updates.


## Literature

1 - [Multi-class brain tumor classification using residual network and global average pooling](https://link.springer.com/article/10.1007/s11042-020-10335-4)