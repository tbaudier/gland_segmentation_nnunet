# Data

## Preperation

The Original data comes from Dicom full body ct scans.

We are faced with two challenges :
- All images must but the same size to go through nnUnet.
- Using the full body of each patient might be a burden for the model as well as the computing power required.

### Find the skull

As we want to segment salivary glands and lacrymal glands, our zone of interest is the head. The goal is to limit our image to the head zone, hence keeping the data we need while reducing the size of it.

The first step used TotalSegmentator ("Tool for segmentation of over 117 classes in CT images") to make a mask of the skull of each patient.

### Resize the skull

To use those skulls properly, we need to get them to the same size as their corresponding CT scan.

See code [here](https://github.com/tbaudier/gland_segmentation_nnunet/blob/main/Data_preparation/resize_skullDataset.py).

### Create the full Dataset

To ensure that our data keeps the same format all the way through the modifications, we change all CT scans and their corresponding skull to a same size. We create (with the same size) label images for supervised learning as well.

See code [here](https://github.com/tbaudier/gland_segmentation_nnunet/blob/main/Data_preparation/createDataset.py).

### Crop the dataset

In order to reduce the amount of data (as well as removing useless information) we crop our images and corresponding labels according to the skull position in space.

See code [here](https://github.com/tbaudier/gland_segmentation_nnunet/blob/main/Data_preparation/croppedDataset.py).

At this point, the dataset is usable for training. We can further reduce the size of our data. However, a drop in quality will occur.

### Small Dataset

Even after the crop, 3 Dimentional data is still quite a heavy format.
We may reduce image size to help with model testing. 

The code [here](https://github.com/tbaudier/gland_segmentation_nnunet/blob/main/Data_preparation/small_Dataset.py) reduces the size by 3 (which is the lowest we recommend before the data dteriorates severly).

## Architecture

```
  data/
  ├──nnUNet_raw
  |  ├── Dataset001_glands
  |  |  ├── imagesTr
  |  |  |  ├── p001_psma_0000.nii.gz
  |  |  |  ├── p002_psma_0000.nii.gz
  |  |  |  └── ...
  |  |  ├── labelsTr
  |  |  |  ├── p001_psma.nii.gz
  |  |  |  ├── p002_psma.nii.gz
  |  |  |  └── ...
  |  |  └── Dataset.json
  |  ├── ..
  |  └── Dataset004_glands
  |     ├── imagesTr
  |     |  ├── p001_psma_0000.nii.gz
  |     |  ├── p002_psma_0000.nii.gz
  |     |  └── ...
  |     ├── labelsTr
  |     |  ├── p001_psma.nii.gz
  |     |  ├── p002_psma.nii.gz
  |     |  └── ...
  |     └── Dataset.json
  ├── nnUNet_preprocessed
  |  ├── Dataset004_glands
  |  |  ├── dataset.json
  |  |  ├── dataset_fingerprint.json
  |  |  ├── nnUNetPlans.json
  |  |  ├── splits_final.json
  |  |  ├── gt_segmentations
  |  |  |  ├── p001_psma.nii.gz
  |  |  |  ├── p002_psma.nii.gz
  |  |  |  └── ..
  |  |  ├── nnUNetPlans_2d
  |  |  |  ├── p001_psma.npy
  |  |  |  ├── p001_psma.npz
  |  |  |  ├── p001_psma.pkl
  |  |  |  ├── p001_psma_seg.npy
  |  |  |  └── ..
  |  |  └── nnUNetPlans_3d_fullres
  |  |     ├── p001_psma.npz
  |  |     ├── p001_psma.pkl
  |  |     └── ..
  |  └── ..
  ├──nnUNet_results
  |  ├── Dataset004_glands
  |  |  ├── nnUNetTrainer_nnUNetPlans_2d
  |  |  |  ├── dataset.json
  |  |  |  ├── dataset_fingerprint.json
  |  |  |  ├── plans.json
  |  |  |  ├── fold_0
  |  |  |  |  ├── checkpoint_best.pth
  |  |  |  |  ├── debug.json
  |  |  |  |  ├── progress.png
  |  |  |  |  ├── training_log_[date+hour+sec].txt
  |  |  |  |  └── ..
  |  |  |  └── ..
  |  |  └── ..
  |  └── ..
  └── Skull_folder
     ├── skull_resized
     |  ├── p001_psma_0000.nii.gz
     |  ├── p002_psma_0000.nii.gz
     |  └── ...
     ├── skull_fullsized
     |  ├── p001_psma_0000.nii.gz
     |  ├── p002_psma_0000.nii.gz
     |  └── ...
     └── skull_cropped
        ├── p001_psma_0000.nii.gz
        ├── p002_psma_0000.nii.gz
        └── ...
``` 
