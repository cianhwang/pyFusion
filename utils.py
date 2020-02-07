#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import numpy as np
from dep2def import depth2defocus
from functools import partial
from awnet import pwc_5x5_sigmoid_bilinear   # cm:import AWnet model
import pytorch_ssim

# In[2]:

AWnet = pwc_5x5_sigmoid_bilinear.pwc_residual().cuda()
AWnet.load_state_dict(torch.load('awnet/fs0_61_294481_0.00919393_dict.pkl'))

width = 256  # img.shape[1]
f = 25
fn = 4
FoV_h = 10 * np.pi / 180
pp = 2 * f * np.tan(FoV_h / 2) / width  # pixel pitch in mm
gamma = 2.4
# use partial is recommended to set lens parameter
myd2d = partial(depth2defocus, f=f, fn=fn, pp=pp, r_step=1, inpaint_occlusion=False)  # this would fix f, fn, pp, and r_step

def t2n(tensor, isImage = True):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    narray = tensor.numpy()
    if isImage:
        if len(narray.shape) == 4:
            narray = narray.transpose(0, 2, 3, 1)
        elif len(narray.shape) == 3:
            narray = narray.transpose(1, 2, 0)
        else:
            raise Exception("convertion error!")
    return narray

def n2t(narray, isImage = True, device = "cuda:0"):
    if isImage:
        if len(narray.shape) == 4:
            narray = narray.transpose(0, 3, 1, 2)
        elif len(narray.shape) == 3:
            narray = narray.transpose(2, 0, 1)  
        else:
            raise Exception("convertion error!")
    tensor = torch.from_numpy(narray).float().to(device)
    return tensor


def get_parameter_number(net):
    '''
    print total and trainable number of params 
    '''
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def dfs_freeze(model):
    '''
    freeze the network
    '''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


# In[3]:


def reconsLoss(J_est, J_gt):   
    '''
    Calculate loss (neg reward) of Reinforcement learning
    
    input: 
        J_est: (B, Seq, C, H, W) predicted image sequences
        J_gt: (B, Seq, C, H, W) ground truth image sequence

    output: 
        lossTensor: (B, 1)
            mse value for each sequence of images in minibatch.
    '''
    lossList = []

    for i in range(J_gt.size()[0]):
        lossList.append(torch.log10(4 / ((J_gt[i] - J_est[i])**2).mean()))
#         lossList.append(F.mse_loss(J_gt[i], J_est[i]))
    lossTensor = torch.stack(lossList)
    #lossTensor = pytorch_ssim.ssim(J_gt/2+0.5, J_est/2+0.5) #torch.stack(lossList)
    return lossTensor

def depth_from_region(depthmap, loc):
    
    H, W = depthmap.shape
    x_l = int((loc[0]+1) * H / 2)
    y_l = int((loc[1]+1) * W / 2)
    x_r = int(min(H, x_l + min(H, W)//4))
    y_r = int(min(W, y_l + min(H, W)//4))
    
    #print("fun_depth_from_region: ({}, {})".format(x_l, y_l))
    
    return depthmap[x_l:x_r, y_l:y_r].mean()

def color_region(tensors, locs):
    
    S, C, H, W = tensors.size()
    
    for i in range(S):
        loc = locs[i]
        x_l = int((loc[0]+1) * H / 2)
        y_l = int((loc[1]+1) * W / 2)
        x_r = int(min(H, x_l + min(H, W)//4))
        y_r = int(min(W, y_l + min(H, W)//4))
        tensors[i][1:, x_l:x_r, y_l:y_r] = -1
        tensors[i][0, x_l:x_r, y_l:y_r] = 1
    
    return tensors

def getDefocuesImage(focusPos, J, dpt):
    '''
    Camera model. 
    Input: 
        focusPos Tensor(B, 1): current timestep focus position [-1, 1]
        J  Tensor (B, C, H, W): next time gt image [0, 1]
        dpt  Tensor (B, 1, H, W): J corresponding depth map [???]
    Output: 
        imageTensor (B, C, H, W): current timestep captured minibatch [0 1]
    '''

    imageTensor = []
    is_cuda_tensor = focusPos.is_cuda
    if is_cuda_tensor:
        focusPos, J, dpt = focusPos.cpu(), J.cpu(), dpt.cpu()

    for i in range(J.size()[0]):
        J_np = t2n(J[i])
        J_np = ((J_np+1)*127.5).astype(np.uint8) # uint8
        dpt_np = dpt[i].squeeze().numpy()*1000
        focusPos_np = focusPos[i].detach().numpy()
        focusPos_np = depth_from_region(dpt_np, focusPos_np)
        focal_img = myd2d(J_np, dpt_np, focusPos_np, inpaint_occlusion=False)
        focal_img = focal_img/127.5-1
        focal_img = n2t(focal_img)
        imageTensor.append(focal_img)
        
    imageTensor = torch.stack(imageTensor)

    if is_cuda_tensor:
        imageTensor = imageTensor.cuda()
    
    return imageTensor

def fuseTwoImages(I, J_hat):
    '''
    AWnet fusion algorithm. 
    Input:
        I Tensor (B, C, H, W): current timestep captured minibatch
        J Tensor (B, C, H, W): last timestep fused minibatch
    Output:
        fusedTensor (B, C, H, W): current timestep fused minibatch
    '''

    with torch.no_grad():
        fusedTensor,_ ,_ = AWnet(J_hat/2+0.5,I/2+0.5)
    
    return torch.clamp(fusedTensor*2-1, -1, 1) 
