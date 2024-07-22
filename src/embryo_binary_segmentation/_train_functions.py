import torch
import csv
from tqdm import tqdm
from time import time
import os
import torchio as tio


def save_losses(epoch, train_loss, val_loss, time, filepath):
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'time':time})

def train(model, device, opt, loss_fn, epochs, data_tr, data_val, save_path, save_each=True, upload_path=None, old_steps=0):
    min_val_loss = float('inf')
    best_epoch = 0
    model_save_path = f"{save_path}/best.model"
    csv_save_path  = f"{save_path}/training_data.csv"

    if upload_path != None:
        if os.path.exists(upload_path):
            model.load_state_dict(torch.load(upload_path))
            print("Loaded model weights from", upload_path)
    
    for epoch_1 in range(epochs):
        epoch = epoch_1 + old_steps
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs+old_steps))

        avg_loss = 0
        model.train() 
        for batch in tqdm(data_tr, desc="Training", leave=True):
            X_batch, Y_batch = batch['image'][tio.DATA].to(device), batch['mask'][tio.DATA].to(device)

            opt.zero_grad()

            Y_pred =  model(X_batch)
            loss = loss_fn(Y_pred, Y_batch) 
            loss.backward()    
            opt.step() 

            avg_loss += loss / len(data_tr)

        toc = time()
        t = toc-tic
        print('Train loss: %f' % avg_loss)          

        val_loss = 0

        if save_each:
            if (epoch + 1) % 5 == 0:
                each_model_save_path = f"{save_path}/epoch_{epoch + 1}.model"
                #each_opt_save_path = f"{save_path}/epoch_{epoch + 1}_optimizer.pt"
                torch.save(model.state_dict(), each_model_save_path)
                #torch.save(opt.state_dict(), each_opt_save_path)
        
        with torch.no_grad():
            for batch in data_val:
                X_val, Y_val = batch['image'][tio.DATA].to(device), batch['mask'][tio.DATA].to(device)
                Y_pred = model(X_val)
                loss = loss_fn(Y_pred, Y_val)
                val_loss += loss/len(data_val)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                print(f"New best model saved with loss {val_loss}")

        save_losses(epoch, float(avg_loss), float(val_loss), float(t), csv_save_path)

    return float(min_val_loss), best_epoch