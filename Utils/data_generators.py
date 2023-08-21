import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import pickle
from torch.utils.data import Dataset, DataLoader 

from .df_utils import get_image_and_label, show_trainimage_from_X
import pdb

class data_loader(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,  image_name_list, target_list):
        'Initialization' 
        self.image_name_list = image_name_list  
        self.target_list = target_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_name_list) 
     

    def read_image(self, imagename):
        f = open(imagename, 'rb')
        img = pickle.load(f,encoding='uint8') / 255.0  
        img = img[:,:,:]  # HxWxC => HxWxC
        img = np.transpose(img, (2, 0, 1)) # CxHxW
        return img.astype('float32') 
    
    def target_array(self, target_value):
        y_  = np.array([target_value]).astype('float32') 
        return y_

    def __getitem__(self, idx):
        #     if torch.is_tensor(idx):
        #         idx = idx.tolist()
         
        img0_name  = self.image_name_list[idx][0]
        img1_name  = self.image_name_list[idx][1] 
         
        
        img0       = self.read_image(img0_name) 
        img1       = self.read_image(img1_name)    
        
        X   = np.concatenate([img0, img1], axis=0)  
        y   = self.target_array(self.target_list[idx])  

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)  

        return X, y


if __name__ == "__main__":
    file = open(os.path.join('/home/gaap-suwichaya/Projects/Solar_Energy/SolarNet/Data', 'Data_3Days.pkl'),'rb')
    new_root = '/home/gaap-suwichaya/Projects/Solar_Energy/SolarNet/Data/SkyImage_3Days/'
    batch_size = 4 
    min_index  = 0
    max_index  = 10
    
    df = pickle.load(file)
    image_name_list, target_list = get_image_and_label(df, new_root, batch_size, min_index, max_index) 
    input_dataset                = data_loader(image_name_list, target_list)
 
    for i, (X, y) in enumerate(input_dataset):
        show_trainimage_from_X(X, image_name= "%d_no-batch.png" % i) 
    
    custom_dataloader = DataLoader(input_dataset, batch_size=batch_size)

    for batch_idx, (batch_X, batch_y) in enumerate(custom_dataloader):
        batch_X
        batch_y
        for i in range(batch_size):  # Plot at most 4 images 
            X_temp = batch_X[i]
            show_trainimage_from_X(X_temp, image_name="%d-%d_batch.png" %(batch_idx, i))
             