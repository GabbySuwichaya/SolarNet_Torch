# from keras.layers import Conv2D, MaxPooling2D
# from keras import Sequential
# from keras import models
# from keras import layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def Conv2D(in_channel=3, out_channel=64, kernel_size=(3,3), stride=1, add_relu=True):
    """1x1 convolution without padding"""
    layers = []  
    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding="same", bias=False))
    if add_relu:
        layers.append(nn.ReLU())
    return layers
    
def Maxpooilng2D(pool_size=(2, 2), stride_touple=(2, 2)):
    return [nn.MaxPool2d(pool_size, stride=stride_touple)]


class BASE(nn.Module):
    def __init__(self, inp_channel=3, num_images=2):
        super().__init__()   

        # def SCNN(input_shape=(256, 256, 3)):
        #   model = Sequential()
        #   model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #   model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #   model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #   model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #   model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        #   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #   return model
 

        self.base = []
        self.add_layer(Conv2D(in_channel=inp_channel*num_images, out_channel = 64, kernel_size=(3,3)))
        self.add_layer(Conv2D(in_channel = 64,  out_channel = 64, kernel_size=(3,3)))
        self.add_layer(Maxpooilng2D(pool_size=(2, 2), stride_touple=(2, 2)))
        self.add_layer(Conv2D(in_channel = 64,  out_channel = 128, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 128, out_channel = 128, kernel_size=(3,3)))
        self.add_layer(Maxpooilng2D(pool_size=(2, 2), stride_touple=(2, 2)))
        self.add_layer(Conv2D(in_channel = 128, out_channel = 256, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 256, out_channel = 256, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 256, out_channel = 256, kernel_size=(3,3)))
        self.add_layer(Maxpooilng2D(pool_size=(2, 2), stride_touple=(2, 2)))
        self.add_layer(Conv2D(in_channel = 256, out_channel = 512, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 512, out_channel = 512, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 512, out_channel = 512, kernel_size=(3,3)))
        self.add_layer(Maxpooilng2D(pool_size=(2, 2), stride_touple=(2, 2)))
        self.add_layer(Conv2D(in_channel = 512, out_channel = 512, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 512, out_channel = 512, kernel_size=(3,3))) 
        self.add_layer(Conv2D(in_channel = 512, out_channel = 512, kernel_size=(3,3)))
        self.add_layer(Maxpooilng2D(pool_size=(2, 2), stride_touple=(2, 2)))

        
    def add_layer(self, new_layer):
        self.base = self.base + new_layer

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor):  
        predict = self.forward_one(x) 
        return predict


class SolarNet(BASE):
    def __init__(self, inp_channel=3, num_images=2):
        BASE.__init__(self, inp_channel=inp_channel, num_images=num_images)
        tops = []
        tops.append(nn.Dropout(0.2))
        tops.append(nn.Linear(512, 256))
        tops.append(nn.Dropout(0.2))
        tops.append(nn.Linear(256, 1))
        
        self.SCNN_base  = nn.Sequential(*self.base)
        self.top_model   = nn.Sequential(*tops)


        self.final   = nn.Linear(64, 1)


    def forward_one(self, x): 

        B,C,H,W = x.shape
        y = self.SCNN_base(x)  
        y = y.view(B, 512,-1) 
        y = y.permute(0,2,1) 
        output = self.top_model(y) 
        output = output.permute(0,2,1)   
        output = self.final(output)
        output = output.view(B,-1) 
        return output





# def CNN3D(input_shape=(256, 256, 2, 3)):
#     model = models.Sequential()
#     model.add(layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape, padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(128, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(128, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(256, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     model.add(layers.Conv3D(256, kernel_size=(3, 3, 2), activation='relu', kernel_initializer='he_uniform', padding='same', data_format = 'channels_last'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
#     return model


