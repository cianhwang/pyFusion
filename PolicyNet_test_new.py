#!/usr/bin/env python
# coding: utf-8

# In[1]:


## test model behavior in synthetic/real images.
## found model output one certain value after training.
## test general mse value

import os
from model import *
from data_loader import *
from utils import *
import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib.pyplot as plt
import toy_utils
import torch.nn.functional as F
from helper_funcs import *
import pytorch_ssim

def load_checkpoint(ckpt_path, model):
    
    ckpt_dir = 'ckpt/'+ckpt_path

    print("[*] Loading model from {}".format(ckpt_dir))

    filename = 'rfc_model_best.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)

    # load variables from checkpoint
    start_epoch = ckpt['epoch']
    best_loss = ckpt['best_valid_mse']
    print("current epoch: {} --- best loss: {}".format(start_epoch, best_loss))
    model.load_state_dict(ckpt['model_state'])
    #optimizer.load_state_dict(ckpt['optim_state'])   

    return model


# In[2]:


model = focusLocNet(0.17, 1, 256, 2).to("cuda:0")


# In[3]:


model = load_checkpoint('0507_16_13', model)


# In[4]:


model = model.eval()


# In[5]:


from awnet import pwc_5x5_sigmoid_bilinear   # cm:import AWnet model
import torch
from torchvision import transforms, utils

AWnet = pwc_5x5_sigmoid_bilinear.pwc_residual().cuda()
AWnet.load_state_dict(torch.load('awnet/fs_34_all_0.03036882.pkl'))
AWnet = AWnet.eval()

import warnings
warnings.filterwarnings("ignore")

def fuseTwoImages(I, J_hat):
    with torch.no_grad():
        fusedTensor,warp,mask = AWnet(J_hat,I)
    return fusedTensor, warp, mask

def patchize(img):
    imgs = []
    H, W, C = img.shape
    ph = H//2
    pw = W//2
    img_empty = np.zeros((H+200, W+200, C))
    img_empty[100:-100, 100:-100] = img
    img = img_empty
    for i in range(2):
        for j in range(2):
            imgs.append(img[100+ph*i-50:100+ph*i+ph+50, 100+pw*j-32:100+pw*j+pw+32])
    imgs = np.stack(imgs)
    return imgs

def depatchize(imgs, pd_h = 50, pd_w = 32):
    ph = (imgs[0].shape[0]-2*pd_h)
    pw = (imgs[0].shape[1]-2*pd_w)
    img = np.zeros((ph*2, pw*2, 3))
    for i in range(2):
        for j in range(2):
            img[i*ph:i*ph+ph, j*pw:j*pw+pw] = imgs[i*2+j, pd_h:-pd_h, pd_w:-pd_w]
            
    return img
    
def image_fuse(a_batch, b_batch):
    batch_size = a_batch.size(0)
    c_batch = []
    for k in range(batch_size):
        a = a_batch[k]
        b = b_batch[k]
        a = a.cpu().detach().numpy().transpose(1, 2, 0) /2. + 0.5
        b = b.cpu().detach().numpy().transpose(1, 2, 0) /2. + 0.5
        
        aa = patchize(a)
        bb = patchize(b)
        aa = torch.Tensor(aa.transpose(0, 3, 1, 2)).cuda()
        bb = torch.Tensor(bb.transpose(0, 3, 1, 2)).cuda()

        ccs = []
        #wws = []
        for i in range(4):
            cc, ww, mask = fuseTwoImages(aa[i:i+1], bb[i:i+1])
            ccs.append(cc[0])
            #wws.append(ww[0])
        cc = torch.stack(ccs)
        #ww = torch.stack(wws)

        c = depatchize(cc.cpu().detach().numpy().transpose(0, 2, 3, 1))
        #warp = depatchize(ww.cpu().detach().numpy().transpose(0, 2, 3, 1))
        c = np.clip(c, 0, 1) * 2 - 1.0
        c = torch.from_numpy(c.transpose(2, 0, 1)).cuda().float()
        c_batch.append(c)
        
    c_batch = torch.stack(c_batch)
    return c_batch


# In[6]:


# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pickle
# import cv2

# class pixel_estimator_with_weights(nn.Module):
#     def __init__(self, Weights,device = "cpu"):
#         ## Default: gpu mode
#         super(pixel_estimator_with_weights, self).__init__()
#         self.device = torch.device(device)
#         self.w1 = torch.from_numpy(Weights[0].transpose(3,2,0,1)).to(self.device)
#         self.b1 = torch.from_numpy(Weights[1]).to(self.device)
#         self.w2 = torch.tensor(Weights[2].transpose(3,2,0,1)).to(self.device)
#         self.b2 = torch.tensor(Weights[3]).to(self.device)
#         self.w3 = torch.tensor(Weights[4].transpose(3,2,0,1)).to(self.device)
#         self.b3 = torch.tensor(Weights[5]).to(self.device)
#         self.w4 = torch.tensor(Weights[6]).reshape(4,4,8,1024).permute(3,2,0,1).to(self.device)
#         self.b4 = torch.tensor(Weights[7]).to(self.device)
#         self.w5 = torch.tensor(Weights[8]).reshape(1,1,1024,512).permute(3,2,0,1).to(self.device)
#         self.b5 = torch.tensor(Weights[9]).to(self.device)
#         self.w6 = torch.tensor(Weights[10]).reshape(1,1,512,10).permute(3,2,0,1).to(self.device)
#         self.b6 = torch.tensor(Weights[11]).to(self.device)
#         self.w7 = torch.tensor(Weights[12]).reshape(1,1,10,1).permute(3,2,0,1).to(self.device)
#         self.b7 = torch.tensor(Weights[13]).to(self.device)

#     def forward(self, x):
#         x = F.relu(F.conv2d(x,self.w1,bias = self.b1,stride=1))
#         x = F.relu(F.conv2d(x,self.w2,bias = self.b2,stride=1,dilation=8))
#         x = F.relu(F.conv2d(x,self.w3,bias = self.b3,stride=1,dilation=32))
#         x = F.leaky_relu(F.conv2d(x,self.w4,bias = self.b4,stride=1,dilation=128),0.1)
#         x = F.leaky_relu(F.conv2d(x,self.w5,bias = self.b5,stride=1),0.1)
#         x = F.leaky_relu(F.conv2d(x,self.w6,bias = self.b6,stride=1),0.1)
#         x = F.conv2d(x,self.w7,bias = self.b7,stride=1)
#         return x
    
# AFmodel = torch.load('autofocus.pth',map_location='cpu')
# AFmodel.eval()    
    
# def crop_patches(img, window= 1023, step = 512):
#     patches = []
#     H, W = img.shape
#     for i in range(0, H-step, step):
#         for j in range(0, W-step, step):
#             patches.append(img[i:i+window, j:j+window])
#     return np.stack(patches)


# def gaf_func(img):
#     assert img.max() <= 1.0
#     assert img.shape == (2160, 3840)
#     img = np.pad(img, ((200, 200), (128, 128)), 'reflect')
#     H, W = img.shape
    
#     patches = crop_patches(img)
#     patches = torch.from_numpy(patches).float().unsqueeze(1)#.cuda()
        
#     results = []
#     with torch.no_grad():
#         for i in range(patches.size()[0]):
#             results.append(AFmodel(patches[i:i+1]))
#     results = torch.stack(results)

#     results = results.numpy()#.cpu()
#     results = results.squeeze()
    
#     k = 0
#     sigma =1
#     n_img = np.zeros((H-512, W-512))
#     for i in range(0, H-512, 512):
#         for j in range(0, W-512, 512):
#             n_img[i:i+512, j:j+512] = results[k]
#             k += 1
    
#     n_img[n_img < 0] = 0
#     n_img = np.clip(n_img, 0, 8)
#     return n_img

# def gaf_func_tensor(Is):
#     batch_size = Is.size(0)
#     afs = []
#     for i in range(batch_size):
#         img = Is[i].cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5
#         img = img.astype(np.uint8)
#         img = cv2.resize(img, (3840, 2160))
#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255.0
#         gaf = gaf_func(gray_img)
#         gaf = cv2.resize(gaf, (3072, 1536))
#         gaf = (gaf-gaf.min())/(gaf.max()-gaf.min())*2.0-1.0
#         gaf = torch.from_numpy(gaf).float().unsqueeze(0).cuda()
# #         print("gaf: ", gaf.size(), gaf.max(), gaf.min())
#         afs.append(gaf)
        
#     afs = torch.stack(afs)
#     return afs


# In[7]:


video_path = "/home/qian/Downloads/DAVIS/test_davis_video_trunc_list.txt"
depth_path = "/home/qian/Downloads/DAVIS/test_davis_dpt_trunc_list.txt"
video_dataset = load_davis_dataset(video_path, depth_path, 2)[0]
# dataiter = iter(video_dataset)


# In[8]:


def reset():
    h = [torch.zeros(1, 1, 256).cuda(),
                  torch.zeros(1, 1, 256).cuda()]
    l = torch.rand(1, 2).cuda()*2.0-1.0 #-0.5~0.5
    return h, l


# In[9]:


import utils_rule
apply_rule = False


# In[ ]:


# x_train, dpt = dataiter.next()
losses = []
losses2 = []
for i, (x_train, dpt) in enumerate(video_dataset):
    if i > 0:
        print(i, sum(losses)/len(losses), sum(losses2)/len(losses2))
    x_train = x_train.cuda()
    dpt = dpt.cuda()
# print(x_train.size())

# # show images
# imshow(torchvision.utils.make_grid(x_train[0]).cpu())

    # print(len(video_dataset))
    # for i, (x_train, dpt) in enumerate(video_dataset):
    #     print(i)

    #     x_train = x_train.cuda()
    #     dpt = dpt.cuda()
    J_est = []
    I_est = []
    afs = []
    locs = []
    mus = []
    J_prev = None
    last_af = None
    h, l = reset()
    if apply_rule:
        l = torch.rand(1, 1).cuda()*2.0 - 1.0
    with torch.no_grad():
        for t in range(x_train.size(1)):
#             print("seq:  ", t)
            if apply_rule:
                I, af, u_in  = utils_rule.getDefocuesImage(l, x_train[:, t, ...], dpt[:, t, ...])
            else:
                I, af, u_in  = getDefocuesImage(l, x_train[:, t, ...], dpt[:, t, ...])

            if J_prev is None:
                J_prev = I
            else:
                ## needs blockwise op
                J_prev = image_fuse(I, J_prev)

    #         print(af.size(), af.max(), af.min())
#             af = gaf_func_tensor(J_prev)
    #         print(af.size(), af.max(), af.min())

            I_est.append(I)
            J_est.append(J_prev)
            
            last_af = af 
            if last_af is None:
                input_t = af
            else:
                input_t = torch.min(af, last_af)


            afs.append(input_t)

            if apply_rule:
                l = utils_rule.rule_based(dpt[:, t, ...], l)
                mu = l
            else:
                h, mu, l, _, _ = model(input_t, l, h)
                l = torch.rand(1, 2).cuda()*2.0-1.0
            locs.append(l)
            mus.append(mu)

        J_est = torch.stack(J_est, dim = 1)
        I_est = torch.stack(I_est, dim = 1)
        afs = torch.stack(afs, dim = 1)
        locs = torch.stack(locs, dim = 1)
        mus = torch.stack(mus, dim = 1)
        losses.append(F.mse_loss(J_est[:,1:], x_train[:,1:]).item())
        losses2.append(F.mse_loss(I_est[:,1:], x_train[:,1:]).item())

# print("mse loss: ", F.mse_loss(J_est[:,1:], x_train[:,1:]).item())
# print("ssim loss: ", pytorch_ssim.ssim(J_est[0,1:], x_train[0,1:]))
# imshow(torchvision.utils.make_grid(color_region(I_est[0], locs[0])).cpu())
# # # imshow(torchvision.utils.make_grid(U_est[0]).cpu())
# imshow(torchvision.utils.make_grid(J_est[0]).cpu())
# imshow(torchvision.utils.make_grid(afs[0]).cpu())
# imshow(torchvision.utils.make_grid(x_train[0]).cpu())


# In[ ]:





# In[ ]:


# import time
# s = int(time.time() % 10000)
# for i in range(4):
#     cv2.imwrite("{}_x_train_{}.jpg".format(s, i), x_train[0, i].cpu().numpy().transpose(1, 2, 0)[..., ::-1] *127.5 + 127.5)
#     cv2.imwrite("{}_J_est_{}.jpg".format(s, i), J_est[0, i].cpu().numpy().transpose(1, 2, 0)[..., ::-1] *127.5 + 127.5)
#     cv2.imwrite("{}_I_est{}.jpg".format(s, i), I_est[0, i].cpu().numpy().transpose(1, 2, 0)[..., ::-1] *127.5 + 127.5)
#     cv2.imwrite("{}_afs{}.jpg".format(s, i), afs[0, i, 0].cpu().numpy()*127.5 + 127.5)
#     cv2.imwrite("{}_afs_w{}.jpg".format(s, i), color_region(afs[0].repeat(1, 3, 1, 1), mus[0])[i].cpu().numpy().transpose(1, 2, 0)[..., ::-1] *127.5 + 127.5)


# In[ ]:




