import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import IO
from skimage import transform
from skimage.measure import regionprops
import torchio as tio
import cv2 as cv
from scipy.ndimage import zoom

def load_images_masks(base_folder):
    images = []
    masks = []

    def recursive_search(folder):
        
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            
            if os.path.isdir(item_path):
                if item.startswith('FUSE'):
                    image_filenames = sorted(os.listdir(item_path))
                    #print(item)
                    print(item_path)
                    #print(image_filenames)
                    for filename in image_filenames:
                        image = IO.imread(os.path.join(item_path, filename))
                        image = np.transpose(image, (2, 0, 1))
                        images.append(image)
                            
                elif item.startswith('SEG'):
                    mask_filenames = sorted(os.listdir(item_path))
                    #print(item)
                    print(item_path)
                    #print(mask_filenames)
                    for filename in mask_filenames:
                        mask = IO.imread(os.path.join(item_path, filename))
                        mask = np.transpose(mask, (2, 0, 1))
                        masks.append(mask)

                
                recursive_search(item_path)

    recursive_search(base_folder)
    
    return images, masks



def cropp(images, masks, binarize=False, target=[64, 512, 512], resize=False):
    
    cropped_images = []
    cropped_masks = []
    
    for image, mask in zip(images, masks):
        
        if binarize:
            binary_mask = np.where(np.array(mask) == 1, 0, 1)
        else:
            binary_mask = np.array(mask)
        
        size = binary_mask.shape

        if resize:
            resize_factors = [max(1, 640 / dim) if dim < 540 else 1 for dim in size]
            if any(factor > 1 for factor in resize_factors):
                image = zoom(image, resize_factors, order=1)  
                binary_mask = zoom(binary_mask, resize_factors, order=0)
                size = binary_mask.shape 
        
        props = regionprops(binary_mask)
        center = props[0].centroid

        cropped_image = np.zeros(target)
        cropped_mask = np.zeros(target)
        
        for i in range(3):
            x = size[i]/2 - center[i]
            left = int((size[i] - target[i])/2 - x)
            right = int((size[i] - target[i])/2 + x + 1)

            if i == 0:
                cropped_mask = binary_mask[left:size[i]-right, :, :]
                cropped_image = image[left:size[i]-right, :, :]
            elif i == 1:
                cropped_mask = cropped_mask[:, left:size[i]-right, :]
                cropped_image = cropped_image[:, left:size[i]-right, :]
            elif i == 2:
                cropped_mask = cropped_mask[:, :, left:size[i]-right]
                cropped_image = cropped_image[:, :, left:size[i]-right]
        
        cropped_images.append(cropped_image)
        cropped_masks.append(cropped_mask)

        #print(f'{size}, {center}, {cropped_image.shape}')
    
    return cropped_images, cropped_masks


def normalization(images):
    cl_images = []
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for image in images:
        z_size = image.shape[0]
        cl_image = np.array([clahe.apply(image[i]) for i in range(z_size)])
        #cl_image = np.array([image[i] for i in range(z_size)])
        min_val = np.min(cl_image)
        max_val = np.max(cl_image)
        cl_image = (cl_image - min_val) / (max_val - min_val)
        
        cl_images.append(cl_image)
    
    return cl_images

def minmaxnormalization(images):
    minmax_images = []
    for image in images:
        
        image = np.array(image)
        min_val = np.min(image)
        max_val = np.max(image)
        minmax_image = (image - min_val) / (max_val - min_val)
        
        minmax_images.append(minmax_image)
    
    return minmax_images
    

class create_dataset(Dataset):
    def __init__(self, image_data_list, mask_data_list, patch_size=None, mode='test'):
        self.image_data_list = image_data_list
        self.mask_data_list = mask_data_list
        self.patch_size = patch_size
        self.mode = mode

        if (self.mode == 'train' or self.mode == 'val'):
            self.subjects = self._extract_subject_patches()

        else:
            self.patches = [{'image': torch.tensor(image_data.astype(np.float32), dtype=torch.float32).unsqueeze(0),
                             'mask': torch.tensor(mask_data.astype(np.float32), dtype=torch.float32).unsqueeze(0)}
                            for image_data, mask_data in zip(self.image_data_list, self.mask_data_list)]

    def __len__(self):
        if hasattr(self, 'subjects'):
            return len(self.subjects)
        return len(self.patches)

    def __getitem__(self, idx):
        if hasattr(self, 'subjects'):
            return self.subjects[idx]
        patch = self.patches[idx]
        image_patch = patch['image']
        mask_patch = patch['mask']
        return image_patch, mask_patch

    def _extract_subject_patches(self):
        subjects = []
        for image_data, mask_data in zip(self.image_data_list, self.mask_data_list):
            z_size, y_size, x_size = image_data.shape
            patch_z, patch_y, patch_x = self.patch_size

            for z in range(0, z_size, patch_z):
                for y in range(0, y_size, patch_y):
                    for x in range(0, x_size, patch_x):
                        image_patch = image_data[z:z+patch_z, y:y+patch_y, x:x+patch_x]
                        image_patch = torch.tensor(image_patch.astype(np.float32), dtype=torch.float32).unsqueeze(0)
                        mask_patch = mask_data[z:z+patch_z, y:y+patch_y, x:x+patch_x]
                        mask_patch = torch.tensor(mask_patch.astype(np.float32), dtype=torch.float32).unsqueeze(0)
                        subject = tio.Subject(
                                    image=tio.ScalarImage(tensor=image_patch),
                                    mask=tio.LabelMap(tensor=mask_patch))
                        subjects.append(subject)

        return subjects



def upload_data(data_path, label, binarize=False, patch_size=[32, 256, 256], target=[64, 512, 512], resize=False):

    image_files, mask_files = load_images_masks(data_path)

    if label == 'train':
        cropped_images, cropped_masks = cropp(image_files, mask_files, binarize, target, resize)
        normalize_images = normalization(cropped_images)
        dataset = create_dataset(normalize_images, cropped_masks, patch_size, 'train')

    elif label == 'val':
        cropped_images, cropped_masks = cropp(image_files, mask_files, binarize, target, resize)
        normalize_images = normalization(cropped_images)
        dataset = create_dataset(normalize_images, cropped_masks, patch_size, 'val')

    elif label == 'test':
        cropped_images, cropped_masks = cropp(image_files, mask_files, binarize, target, resize)
        normalize_images = normalization(cropped_images)
        dataset = create_dataset(normalize_images, cropped_masks, 'test')

    return dataset 