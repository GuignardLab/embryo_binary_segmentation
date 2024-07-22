import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import torchio as tio


def predict_with_images(model, data, model_path, device):
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    predictions = []
    images = []
    masks = []

    for batch in data:
        X_batch, Y_batch = batch['image'][tio.DATA].to(device), batch['mask'][tio.DATA].to(device)
        Y_pred = torch.squeeze(model(X_batch.to(device)).detach())
        Y_pred = Y_pred[0, :, :, :]
        X_batch = X_batch[0, :, :, :, :]
        X_batch = X_batch[0, :, :, :]
        Y_batch = Y_batch[0, :, :, :, :]
        Y_batch = Y_batch[0, :, :, :]
        predictions.append(Y_pred)
        images.append(X_batch)
        masks.append(Y_batch)
        
    return predictions, images, masks


def visualize_random_pred(image_files, mask_files, preds, num_slices=4, binary=True, threshold=0.5, alpha=0.5):
    random_indices = random.sample(range(len(image_files)), min(num_slices, len(image_files)))

    plt.figure(figsize=(10, 2 * num_slices)) 

    for i, idx in enumerate(random_indices):
        image = image_files[idx].cpu().detach().numpy()
        mask = mask_files[idx].cpu().detach().numpy()
        pred = preds[idx].cpu().detach().numpy()

        z = random.randint(0, image.shape[0] - 1)

        plt.subplot(num_slices, 5, 5 * i + 1)
        plt.imshow(image[z], cmap='gray')
        plt.title(f'Image Slice {z}, patch {i}')
        plt.axis('off')

        plt.subplot(num_slices, 5, 5 * i + 2)
        if binary:
            pred = np.where(pred < threshold, 0, 1)
        plt.imshow(pred[z], cmap='binary')
        plt.title(f'Prediction Slice {z}')
        plt.axis('off')

        plt.subplot(num_slices, 5, 5 * i + 3)
        overlay_mask = np.zeros_like(image[z], dtype=np.float32)
        overlay_mask[pred[z] > 0] = threshold
        plt.imshow(image[z], cmap='gray')
        plt.imshow(overlay_mask, cmap='cool', alpha=alpha)
        plt.title(f'Prediction Overlay {z}')
        plt.axis('off')

        plt.subplot(num_slices, 5, 5 * i + 4)
        plt.imshow(mask[z], cmap='binary')
        plt.title(f'Mask Slice {z}, patch {i}')
        plt.axis('off')

        plt.subplot(num_slices, 5, 5 * i + 5)
        overlay_mask = np.zeros_like(image[z], dtype=np.float32)
        overlay_mask[mask[z] > 0] = 0.5
        plt.imshow(image[z], cmap='gray')
        plt.imshow(overlay_mask, cmap='cool', alpha=alpha)
        plt.title(f'Mask Overlay {z}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



def score_model(model, model_path, metric, data, device, threshold_pixel = 0.5):
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    scores = 0
    for batch in data:
        X_batch, Y_label = batch['image'][tio.DATA].to(device), batch['mask'][tio.DATA].to(device)
        Y_pred = torch.squeeze(model(X_batch.detach())) > threshold_pixel 
        Y_pred = Y_pred.long()
        score = metric(Y_pred, Y_label).mean().item()
        #print("Score: ", score, "\n")
        scores += score

    return scores/len(data)


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1).byte() 
    labels = labels.squeeze(1).byte()
    #print(f'Output Shape {outputs.shape}, Labels Shape {labels.shape}')

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2, 3))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  

    return iou


def plot_loss_and_time(csv_path, title, save_path=None):
    epochs = []
    train_losses = []
    val_losses = []
    times = []

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            epoch, train_loss, val_loss, time = map(float, row)
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            times.append(time)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    cumulative_time = [sum(times[:i+1])/60 for i in range(len(times))]
    plt.subplot(1, 2, 2)
    plt.plot(epochs, cumulative_time)
    plt.ylabel('Time (minutes)')
    plt.xlabel('Epoch')
    plt.title('Cumulative Time per Epoch')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png')
        
    plt.show()
