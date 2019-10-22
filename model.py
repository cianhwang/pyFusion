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
        
        self.block1 = convBlock(3, 16, 7, 2)
        self.block2 = convBlock(16, 32, 5, 2)
        self.block3 = convBlock(32, 64, 5, 2)
        self.block4 = convBlock(64, 64, 5, 2)
        self.block5 = convBlock(64, 128, 5, 2)        
        self.block6 = convBlock(128, 128, 5, 4, isBn = False)
        self.lstm = nn.LSTMCell(2304, 512)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)   
        
        self.lstm_hidden = self.init_hidden()
        
    def init_hidden(self):
        self.lstm_hidden = None
        return
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.block4(x) 
        x = self.block5(x) 
        x = self.block6(x)
        
        x = x.view(x.size()[0], -1)
        
        if self.lstm_hidden is None:
            self.lstm_hidden = self.lstm(x)

        else:
            self.lstm_hidden = self.lstm(x, self.lstm_hidden)

#             self.h, self.c = self.lstm(x, (self.h, self.c))
        x = F.relu(self.lstm_hidden[0])
#         x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = torch.tanh(self.fc4(x))
        
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        pos = mu + noise

        # bound between [-1, 1]
        pos = torch.tanh(pos)
        
        log_pi = Normal(mu, self.std).log_prob(pos)
        log_pi = torch.sum(log_pi, dim=1)
        
        return mu, pos, log_pi

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

