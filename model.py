#!/usr/bin/env python
# coding: utf-8

# # ROBOTICS FOCUS CONTROL

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# In[4]:


class focusLocNet(nn.Module):
    '''
    Description: analyze estimated ^J_{t-1} to get next focus position sampled from Gaussian distr.
    
    input: 
        x: (B, 3, 512, 896) image tensor
            range [-1, 1]

    output: 
        mu: (B, 1) mean of gaussian distribution
            range [-1, 1]
        pos: (B, 1) normalized focus position
            range [-1, 1]
        log_pi: logarithmatic probabilty of choosing pos ~ Gauss(mu, self.std)
        
    arguments:
        std: std of gaussian distribution
            
    '''
    
    def __init__(self, std = 0.17):
        super(focusLocNet, self).__init__()
        
        self.std = std
        
        self.block1 = convBlock(3, 16, 5, 2)
        self.block2 = convBlock(16, 32, 5, 2)
        self.block3 = convBlock(32, 32, 5, 2)
        self.block4 = convBlock(32, 64, 3, 2, isBn = False)
        self.fc0 = nn.Linear(2, 16)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256+16, 256)
        self.fc3 = nn.Linear(256, 256)
        self.lstm = nn.LSTMCell(256, 128)

        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 2) 
        
        self.fc6 = nn.Linear(128, 1)
        
        self.init_hidden()
        
    def init_hidden(self):
        self.lstm_hidden = None
        return
        
    def forward(self, x, l_prev):
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.block4(x) 
#         x = self.block5(x) 
#         x = self.block6(x)
        
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc0(l_prev))
        x = torch.cat((x, y), dim = 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        if self.lstm_hidden is None:
            self.lstm_hidden = self.lstm(x)
        else:
            self.lstm_hidden = self.lstm(x, self.lstm_hidden)

#             self.h, self.c = self.lstm(x, (self.h, self.c))
        x = F.relu(self.lstm_hidden[0])
        b = self.fc6(x.detach()).squeeze(1)
#         x = F.leaky_relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        mu = torch.tanh(self.fc5(x))
        
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        pos = mu + noise

        # bound between [-1, 1]
        pos = torch.tanh(pos)

        log_pi = Normal(mu, self.std).log_prob(pos)
        log_pi = torch.sum(log_pi, dim=1)
         
        return mu, pos, b, log_pi

class convBlock(nn.Module):
    '''
    Conv+ReLU+BN
    '''

    def __init__(self, in_feature, out_feature, filter_size, stride = 1, activation = F.relu, isBn = True):
        super(convBlock, self).__init__()
        self.isBn = isBn
        self.activation = activation

        self.conv1 = nn.Conv2d(in_feature, out_feature, filter_size, stride=stride)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(out_feature)

    def forward(self, x):
        x = self.conv1(x)

        if self.activation is not None:
            x = self.activation(x)        
            
        if self.isBn:
            x = self.bn1(x)
        return x            

class RandomPolicy(object):
    def __init__(self):
        pass
    def __call__(self, loc):
        loc_n = torch.rand_like(loc) * 2 - 1
        if loc_n.is_cuda:
            loc_n = loc_n.cuda()
        return loc_n
        
class CentralPolicy(object):
    def __init__(self):
        pass
    def __call__(self, loc):
        loc_n = torch.zeros_like(loc)
        if loc_n.is_cuda:
            loc_n = loc_n.cuda()
        return loc_n


