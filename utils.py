#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import numpy as np
from dep2def import depth2defocus
from functools import partial
from awnet import pwc_5x5_sigmoid_bilinear   # cm:import AWnet model


# In[2]:

AWnet = pwc_5x5_sigmoid_bilinear.pwc_residual().cuda()
AWnet.load_state_dict(torch.load('awnet/fs0_61_294481_0.00919393_dict.pkl'))


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
        lossList.append(F.mse_loss(J_gt, J_est))
    
    lossTensor = torch.stack(lossList)
    return lossTensor
   
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
    width = 1080  # img.shape[1]
    f = 25
    fn = 4
    FoV_h = 10 * np.pi / 180
    pp = 2 * f * np.tan(FoV_h / 2) / width  # pixel pitch in mm
    gamma = 2.4
    # use partial is recommended to set lens parameter
    myd2d = partial(depth2defocus, f=f, fn=fn, pp=pp)  # this would fix f, fn, pp, and r_step
    imageTensor = []
    is_cuda_tensor = focusPos.is_cuda
    if is_cuda_tensor:
        focusPos, J, dpt = focusPos.cpu(), J.cpu(), dpt.cpu()

    for i in range(J.size()[0]):
        J_np = J[i].numpy().transpose(1, 2, 0)
        dpt_np = dpt[i].squeeze().numpy()
        focusPos_np = focusPos[i].squeeze().detach().numpy()/2+0.5
        #focal_img = myd2d(J_np, dpt_np, focusPos_np, inpaint_occlusion=True)
        focal_img = torch.rand_like(J[i])
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
        fusedTensor,_ ,_ = AWnet(J_hat,I)
    
    return fusedTensor 
