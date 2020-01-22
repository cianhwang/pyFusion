#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[2]:


import h5py
import torch.utils.data as data
import numpy as np
import torch
import os
import os.path
from skimage import transform
from skimage.io import imread
import torch.utils.data
import torchvision
import re

def make_dataset(list_name, dpt_list_name):
    text_file = open(list_name, 'r')
    images_list = text_file.readlines()
    text_file.close()
    images_list = [os.path.join(os.getcwd(), i) for i in images_list]
    #print(images_list)
    
    text_file = open(dpt_list_name, 'r')
    dpt_list = text_file.readlines()
    text_file.close()
    dpt_list = [os.path.join(os.getcwd(), i) for i in dpt_list]
    #print(dpt_list)
    
    return images_list, dpt_list

class DAVISImageFolder(data.Dataset):

    def __init__(self, list_path, dpt_list_path, seq):
        img_list, dpt_list = make_dataset(list_path, dpt_list_path)
        if len(img_list) == 0:
            raise RuntimeError('Found 0 images in: ' + list_path)
        self.list_path = list_path
        self.img_list = img_list
        
        self.dpt_list_path = dpt_list_path
        self.dpt_list = dpt_list

        self.resized_height = 64
        self.resized_width = 128

        self.seq = seq

    def load_imgs(self, img_path):
        img = imread(img_path)
        img = np.float32(img)/127.5-1
        img = transform.resize(img, (self.resized_height, self.resized_width))
        img = np.clip(img, -1.0, 1.0)
        return img
    
    def load_dpt(self, dpt_path):
        dpt = np.load(dpt_path)
        dpt = transform.resize(dpt, (self.resized_height, self.resized_width))

        return dpt

    def __getitem__(self, index):
        
        imgSeq = []
        dptSeq = []

#         ## check border...
#         curr_string = self.img_list[index]
#         if index+self.seq > len(self.img_list):
#             return imgSeq, dptSeq
#         next_string = self.img_list[index+self.seq-1]
#         idx_curr = int(re.findall(r'\d+', curr_string)[0])
#         idx_next = int(re.findall(r'\d+', next_string)[0])
#         if idx_next - idx_curr != self.seq-1:
#             return imgSeq, dptSeq
        
        for i in range(self.seq):
            h5_path = self.img_list[min(index+i, len(self.img_list)-1)].rstrip()
            img = self.load_imgs(h5_path)
            final_img = torch.from_numpy(np.ascontiguousarray(
                img).transpose(2, 0, 1)).contiguous().float()
            imgSeq.append(final_img)
        
        imgSeq = torch.stack(imgSeq)
        
        for i in range(self.seq):
            dpt_h5_path = self.dpt_list[min(index+i, len(self.img_list)-1)].rstrip()
            dpt = self.load_dpt(dpt_h5_path)
            final_dpt = torch.from_numpy(np.ascontiguousarray(
                dpt)).unsqueeze(0).contiguous().float()
            dptSeq.append(final_dpt)
        dptSeq = torch.stack(dptSeq)
        
        return imgSeq, dptSeq

    def __len__(self):
        return len(self.img_list)
    
    
class DAVISDataLoader():
    def __init__(self, list_path, dpt_list_path, seq, _batch_size):
        dataset = DAVISImageFolder(list_path=list_path, dpt_list_path=dpt_list_path, seq = seq)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=True,
                                                       num_workers=int(1))
        self.dataset = dataset

    def load_data(self):

        return self.data_loader

    def __len__(self):

        return len(self.dataset)


# In[3]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(20,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_davis_dataset(seq=3, batch_size=1):
    video_list = "../datasets/DAVIS/test_davis_video_sublist.txt"
    dpt_list = "../datasets/DAVIS/test_davis_dpt_sublist.txt"
    video_data_loader = DAVISDataLoader(video_list, dpt_list, seq, batch_size)
    video_dataset = video_data_loader.load_data()
    return video_dataset

if __name__ == '__main__':
    video_dataset = load_davis_dataset()
    for i, data in enumerate(video_dataset):
        print("index:", i)
        imgSeq = data[0]
        dptSeq = data[1]
        imshow(torchvision.utils.make_grid(imgSeq[0]))
        imshow(torchvision.utils.make_grid(dptSeq[0]/5))
        break

