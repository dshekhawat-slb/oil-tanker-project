#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


# In[ ]:





# In[73]:
import os
from os.path import exists
import cv2

class CustomGen(keras.utils.Sequence):
    
    def __init__(self, img_path, img_path_mask, batch_size):
        self.img_path = img_path
        self.img_path_mask = img_path_mask
        self.batch_size = batch_size

    def __len__(self):
        list_elements = os.listdir(self.img_path)
        return len(self.list_elements) // self.batch_size
    
    def __getitem__(self,idx):
        
        X_paths = self.list_elements[idx*self.batch_size:(idx+1)*self.batch_size]
        y_paths = self.list_elements_mask[idx*self.batch_size:(idx+1)*self.batch_size]
        
        X = []
        y = []
        
        for x_filename in X_paths:
            X.append(load_img(self.img_path + x_filename))
            y_filename = x_filename.replace('.jpg','_mask.jpg')
            y_path = self.img_path_mask + y_filename
            file_exists = exists(y_path)
            if file_exists:
                # y.append(load_img('../data/Oil Tanks/image_patches/'+x_filename))
                y.append(load_img(y_path))
            else:
                img = cv2.imread(self.img_path + x_filename)
                black_img = img*0[img.shape(0), img.shape(1), 0:1]
                y.append(black_img)
        
        return (X,y)


# In[74]:


path = '../data/Oil Tanks/image_patches/'
path_data = '../Oil Tanks/'
path_input = path_data + 'image_patches'
path_mask = path_data + 'image_patches_mask'

# In[75]:


import os


# In[90]:


batch_size = 8
idx = 1
print(f'{idx*batch_size}:{(idx+1)*batch_size}')


# In[76]:


l_input = os.listdir(path_input)

l_output = os.listdir(path_mask)


# In[77]:


l_test = [1,2,4,3,2,4,5,3,2,1,3,4,5,3]
l_test_mask = [2,4,5,3,2,4,5,3,2,1,3,4,5,3]


# In[78]:


data_loader = CustomGen(l_input,l_input,16)


# In[79]:


l_test[1* 2:(1+1)*2]


# In[80]:


len(data_loader)


# In[87]:


x,y = data_loader.__getitem__(1)


# In[88]:


x


# In[83]:


len(data_loader)


# In[72]:


for i in range(len(data_loader)):
    x,y = data_loader.__getitem__(i)
    print(x.shape,y.shape)


# In[ ]:


data_loader

