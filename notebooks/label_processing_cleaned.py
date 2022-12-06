#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# get_ipython().system('pwd')


# In[2]:


path_data = '../Oil Tanks/'


# In[112]:


import json

json_coco = json.load(open(path_data + 'labels_coco.json'))
# json_coco = json.load("../../Oil Tanks/")


# In[4]:


json_coco.keys()


# In[5]:


json_coco


# In[34]:
'''#################################################

import pandas as pd
df = pd.DataFrame(json_coco['annotations'])
df = df[df['image_id']==39]
df


# In[7]:


#import dependencies
import cv2

#Read images
img = cv2.imread(path_data+'image_patches/01_2_8.jpg')


# In[11]:


img.shape


# In[10]:


#Display image
import matplotlib.pyplot as plt
plt.imshow(img)


# In[24]:


imageline = img.copy()
first_label = json_coco['annotations'][0]
pointA = (first_label['bbox'][0], first_label['bbox'][1])
width = first_label['bbox'][2]
height = first_label['bbox'][3]
pointB = (pointA[0]+ width, pointA[1])
pointC = (pointB[0], pointB[1]+height)
pointD = (pointC[0] -width, pointC[1])
pointA, pointB, pointC, pointD


# In[25]:


color = (255, 0, 0)
thickness = 2
img_attonated = cv2.rectangle(imageline, pointA, pointC, color, thickness)


# In[26]:


plt.imshow(img_attonated)


# In[35]:


mapping = {x['id']:x['file_name'] for x in json_coco['images'] }
mapping[39]


# In[36]:


img39 = cv2.imread(path_data+'image_patches/01_3_9.jpg')


# In[39]:


json_coco.keys()


# In[42]:


json_coco['categories']


# In[113]:


json_coco['annotations']
# len(set(k['image_id'] for k in json_coco['annotations']))


# In[51]:


json_coco['images']
# len(set(k['id'] for k in json_coco['images']))


# In[53]:

'''
def get_file(id):
    image_mapping ={x['id']:x['file_name'] for x in json_coco['images']}
    return image_mapping[id]

get_file(4)


# In[96]:

# annotation mapping
annotations_mapping = {}
# x = json_coco['annotations'][28]
# x
for x in json_coco['annotations']:
    key = x ['image_id']
    bbox = x['bbox']
    if key not in annotations_mapping:
        annotations_mapping[key] = []
    annotations_mapping[key].append(bbox)


# In[104]:





# In[105]:


annotations_mapping[key]


# In[114]:

def get_annotation(id):
    annotations_mapping = {}
    for x in json_coco['annotations']:
        key=x['image_id']
        bbox=x['bbox']

        if key not in annotations_mapping:
            annotations_mapping[key] = []

        annotations_mapping[key].append(bbox)
    return annotations_mapping[id]

# annotations_mapping(5)

# In[123]:


[[key,value] for key,value in annotations_mapping.items()]


# In[118]:


pd.DataFrame(annotations_mapping)


# In[68]:


def get_annotations(id):
    try:
        annotations_mapping = {x['image_id']:x['bbox'] for x in json_coco['annotations']}
        annotations = annotations_mapping[id]
    except:
        print("No bbox")
    return annotations
get_annotations(5)


# In[ ]:


# mapping_bbox = {x['id']:x[''] for x in json_coco['annotations'] }
# mapping[39]
import cv2

# def get_bbox_coordinates(id):
#     file = get_file(id)
#     img_original = cv2.imread(path_data + file)
#     first_label =

