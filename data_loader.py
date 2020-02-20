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
import skimage.transform
from skimage.io import imread
import torch.utils.data
import torchvision
from torchvision import transforms
import re
import torchvision.transforms.functional as TF
import random
import numbers
from PIL import Image
import collections
import sys
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

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

def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class SeqToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic_seq):
        for i, pic in enumerate(pic_seq):
            pic_seq[i] = TF.to_pil_image(pic, self.mode)
        return pic_seq
    
class SeqRandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img_seq):
        randnum = random.random()
        for i, img in enumerate(img_seq):
            if randnum < self.p:
                img_seq[i] = TF.hflip(img)
        return img_seq
    
class SeqRandomVerticalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img_seq):
        randnum = random.random()
        for i, img in enumerate(img_seq):
            if randnum < self.p:
                img_seq[i] = TF.vflip(img)
        return img_seq
    
class SeqRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_seq):

        i, j, h, w = self.get_params(img_seq[0], self.size)

        for i, img in enumerate(img_seq):
            img_seq[i] = TF.crop(img, i, j, h, w)
        return img_seq
    
class SeqResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_seq):
        for i, img in enumerate(img_seq):
            img_seq[i] = TF.resize(img, self.size, self.interpolation)
        return img_seq
    
class SeqToTensor(object):
    def __call__(self, img_seq):
        for i, img in enumerate(img_seq):
            img_seq[i] = TF.to_tensor(img)
        return torch.stack(img_seq)
    

class DAVISImageFolder(data.Dataset):

    def __init__(self, list_path, dpt_list_path, seq, transform = transforms.Compose([
                                SeqToPILImage(),
                                SeqRandomHorizontalFlip(),
                                SeqRandomVerticalFlip(),
                                SeqRandomCrop((448, 832)),
                                SeqResize((64, 128)),
                                SeqToTensor()])):
        img_list, dpt_list = make_dataset(list_path, dpt_list_path)
        if len(img_list) == 0:
            raise RuntimeError('Found 0 images in: ' + list_path)
        self.img_list = img_list
        self.dpt_list = dpt_list

#         self.resized_height = 64
#         self.resized_width = 128

        self.seq = seq
        self.transform = transform
        self.offset = 2
        
    def load_imgs(self, img_path):
        img = imread(img_path)
#         img = np.float32(img)/127.5-1
#         #img = transform.resize(img, (self.resized_height, self.resized_width))
#         img = np.clip(img, -1.0, 1.0)
        return img
    
    def load_dpt(self, dpt_path):
        dpt = np.load(dpt_path)
        dpt = np.float32(skimage.transform.resize(dpt, (480, 854)))

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
            h5_path = self.img_list[index].rstrip()
            dpt_h5_path = self.dpt_list[index].rstrip()
            
            try:
                h5_path_offset = self.replace_num_in_string(h5_path, i*self.offset)
                dpt_h5_path_offset = self.replace_num_in_string(dpt_h5_path, i*self.offset)
                img = self.load_imgs(h5_path_offset)
                dpt = self.load_dpt(dpt_h5_path_offset)
            except FileNotFoundError:
                img = self.load_imgs(h5_path)
                dpt = self.load_dpt(dpt_h5_path)
            imgSeq.append(img)
            dptSeq.append(dpt)
            
#         imgSeq = torch.stack(imgSeq)
#         dptSeq = torch.stack(dptSeq)
        imgSeq = self.transform(imgSeq)
        dptSeq = self.transform(dptSeq)
        imgSeq = torch.clamp(imgSeq*2.0-1.0, -1.0, 1.0)
        
        return imgSeq, dptSeq

    def __len__(self):
        return len(self.img_list)

    def replace_num_in_string(self, string, offset):
        if offset == 0:
            return string
        regex = r"00[0-9][0-9][0-9]"
        matches = re.search(regex, string)
        try:
            matches.group()
        except:
            print(string)
        subst = "{:05d}".format(int(matches.group()) + offset)
        result = re.sub(regex, subst, string, 0, re.MULTILINE)
        return result

        
        
    
    
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

def load_davis_dataset(video_path = "../datasets/DAVIS/test_davis_video_sublist.txt", depth_path = "../datasets/DAVIS/test_davis_dpt_sublist.txt", seq=3, batch_size = 1):
    video_list = video_path
    dpt_list = depth_path
    video_data_loader = DAVISDataLoader(video_list, dpt_list, seq, batch_size)
    video_dataset = video_data_loader.load_data()
    return video_dataset

if __name__ == '__main__':
    video_dataset = load_davis_dataset()
    for i, data in enumerate(video_dataset):
        print("index:", i)
        imgSeq = data[0]
        print(imgSeq.shape, imgSeq.max(), imgSeq.min())
        dptSeq = data[1]
        print(dptSeq.shape, dptSeq.max(), dptSeq.min())
#        imshow(torchvision.utils.make_grid(imgSeq[0]))
#        imshow(torchvision.utils.make_grid(dptSeq[0]/5))
#        break

