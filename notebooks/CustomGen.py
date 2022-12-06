## Data Loader definition
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from os.path import exists
import cv2

from tensorflow.keras import layers, Model
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import ast
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import callbacks


from tensorflow.image import resize
from tensorflow.keras import utils


from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model

class CustomGen(keras.utils.Sequence):

    def __init__(self, img_path, img_path_mask, batch_size):
        self.img_path = img_path
        self.img_path_mask = img_path_mask
        self.batch_size = batch_size
        self.list_elements = os.listdir(self.img_path)
        self.list_elements_mask = os.listdir(self.img_path_mask)
        #print(self.__dict__)

    def __len__(self):
#         self.list_elements = os.listdir(self.img_path)
#         self.list_elements_mask = os.listdir(self.img_path_mask)
        return len(self.list_elements) // self.batch_size

    def __getitem__(self,idx):        
        X_paths = self.list_elements[idx * self.batch_size:(idx+1) * self.batch_size]
        y_paths = self.list_elements_mask[idx * self.batch_size:(idx+1) * self.batch_size]

        X = []
        y = []

        for x_filename in X_paths:
            img = np.array(utils.load_img(self.img_path + x_filename,
                                 grayscale=False,
                                 color_mode='rgb',
                                 target_size=(256,256),
                                 interpolation='nearest'))


            X.append(img)
            y_filename = x_filename.replace('.jpg','_mask.jpg')
            y_path = self.img_path_mask + y_filename
            file_exists = exists(y_path)
            if file_exists:
                # y.append(load_img('../data/Oil Tanks/image_patches/'+x_filename))
                black_img = np.array(utils.load_img(y_path,
                                 grayscale=False,
                                 color_mode='rgb',
                                 target_size=(256,256),
                                 interpolation='nearest'))
                image_black_resized = black_img[:, :, 0:1]/255.

                image_black_resized[image_black_resized>0.5]=1.
                image_black_resized[image_black_resized<0.5]=0.              
                
                
            else:
#                 import ipdb; ipdb.set_trace()
                image_black_resized = (img*0)[:, :, 0:1]

            

            y.append(image_black_resized)
        return np.stack(X)/255., np.stack(y), X_paths