import pandas as pd
import os, sys, pickle 

from Utils.model import SolarNet as SCNN  

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from tqdm import tqdm

import pickle
from Utils.df_utils import get_image_and_label
from Utils.data_generators import data_loader

import pdb

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
else:
    device = torch.device("cpu") 

file = open(os.path.join('/home/gaap-suwichaya/Projects/Solar_Energy/SolarNet/Data', 'Data_3Days.pkl'),'rb')
 

new_root = '/home/gaap-suwichaya/Projects/Solar_Energy/SolarNet/Data/SkyImage_3Days/'
batch_size = 16 
min_index  = 0
max_index  = None


df = pickle.load(file)
image_name_list, target_list = get_image_and_label(df, new_root, batch_size, min_index, max_index) 
 
input_dataset                = data_loader(image_name_list, target_list)

file.close() 
  
num_images  = 2 
inp_channel = 3  
train = True

if train:
    model = SCNN(inp_channel=inp_channel, num_images=num_images).to(device)
    

    num_epochs = 60
    # Define loss function and optimizer
    criterion = nn.L1Loss()  # Mean Absolute Error (MAE) loss
    optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.0, amsgrad=False)
    # TF version: optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)

    custom_dataloader = DataLoader(input_dataset, batch_size=batch_size)
    pbar = tqdm(total=len(custom_dataloader)*num_epochs, leave=True)
    
    for epochs in range(num_epochs):
        for batch_idx, (batch_X, batch_y) in enumerate(custom_dataloader):

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            model.train()
            optimizer.zero_grad()

            y = model(batch_X)
        
            loss = criterion(y, batch_y)
            loss.backward()
            optimizer.step()
        
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()        

        if (epochs + 1) % 30 == 0 :
            checkpoint_path = f"Weights_EP{epochs + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    

checkpoint_path = "Weights_EP60.pth"  # Specify the path to the saved checkpoint
loaded_model = SCNN(inp_channel=inp_channel, num_images=num_images).to(device) # Create an instance of the model
loaded_model.load_state_dict(torch.load(checkpoint_path))  # Load the saved state dictionary
loaded_model.eval() 

custom_dataloader = DataLoader(input_dataset, batch_size=1, shuffle=True) 
total_mae = 0
num_samples = 0

predictions_list = []
target_list = []
for  batch_idx, (batch_X, batch_y) in enumerate(custom_dataloader):
    data_batch, target_batch = batch_X.to(device), batch_y.to(device)
    with torch.no_grad():
        predictions = loaded_model(data_batch)

    absolute_diff = torch.abs(predictions - target_batch)
    
    total_mae += absolute_diff.sum().item()
    num_samples += data_batch.size(0) 
    predictions_list.append(predictions.item())

    target_list.append(target_batch.item())

mae = total_mae / num_samples
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(predictions_list)
print(target_list)