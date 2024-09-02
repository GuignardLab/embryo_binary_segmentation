import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
import IO
from tifffile import imsave

def resize_to_multiple_of_4(image):
    z, x, y = image.shape
    new_x = (x + 7) // 8 * 8
    new_y = (y + 7) // 8 * 8
    new_z = (z + 7) // 8 * 8

    padded_image = np.zeros((new_z, new_x, new_y), dtype=image.dtype)
    padded_image[:z, :x, :y] = image
    
    return padded_image, (z, x, y)


def resize_back_to_original(prediction, original_size):
    original_x, original_y, original_z = original_size
    z, x, y = prediction.shape

    unpadded_prediction = prediction[:original_z, :original_x, :original_y]
    
    return unpadded_prediction

def load_images(base_folder):
    images = []
    image_filenames = []
    original_sizes = []

    def recursive_search(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            
            if os.path.isdir(item_path):
                if item.startswith('FUSE'):
                    filenames = sorted(os.listdir(item_path))
                    print(item_path)
                    for filename in filenames:
                        image = IO.imread(os.path.join(item_path, filename))
                        image = np.transpose(image, (2, 0, 1))
                        resized_image, original_size = resize_to_multiple_of_4(image)
                        images.append(resized_image)
                        image_filenames.append(filename)
                        original_sizes.append(original_size)
                recursive_search(item_path)

    recursive_search(base_folder)
    
    return images, image_filenames, original_sizes

def normalization_2(images):
    cl_images = []
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for image in images:
        z_size = image.shape[0]
        cl_image = np.array([clahe.apply(image[i]) for i in range(z_size)])
        min_val = np.min(cl_image)
        max_val = np.max(cl_image)
        cl_image = (cl_image - min_val) / (max_val - min_val)
        cl_images.append(cl_image)
    return cl_images

def generate_patches(images):
    patches = [{'image': torch.tensor(image_data.astype(np.float32), dtype=torch.float32).unsqueeze(0)} for image_data in images]
    return patches

def save_predictions(predictions, filenames, original_sizes, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for prediction, filename, original_size in zip(predictions, filenames, original_sizes):
        resized_prediction = resize_back_to_original(prediction, original_size)
        binary_prediction = (resized_prediction > 0.5).astype(np.uint8) 
        save_path = os.path.join(output_folder, f"{filename}_pred.tif")
        imsave(save_path, binary_prediction)



def made_predictions(base_folder, model, output_folder, batch_size, device):
    images, image_filenames, original_sizes = load_images(base_folder)
    images = normalization_2(images)
    patches = generate_patches(images)
    predictions_loader = DataLoader(patches, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in predictions_loader:
            inputs = batch['image'].to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    save_predictions(predictions, image_filenames, original_sizes, output_folder)