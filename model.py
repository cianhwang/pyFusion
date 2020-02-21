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
    
    def __init__(self, std, hidden_size):
        super(focusLocNet, self).__init__()
        
        self.std = std
        
        self.block1 = convBlock(3, 16, 5, 2)
        self.block2 = convBlock(16, 32, 5, 2)
        self.block3 = convBlock(32, 32, 5, 2)
        self.block4 = convBlock(32, 64, 3, 2, isBn = False)
        self.fc0 = nn.Linear(2, 16)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256+16, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1)

        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5_0 = nn.Linear(128, 128)
        self.bn5_0 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 2)

        
        self.fc6_0 = nn.Linear(128, 128)
        self.bn6_0 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 1)
        
        self.hidden_size = hidden_size
        
    def forward(self, x, l_prev, h_prev):
        batch_size = x.size(0)
        
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.block4(x) 
        
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc0(l_prev))
        x = torch.cat((x, y), dim = 1)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        x, hidden = self.lstm(x.view(1, *x.size()), h_prev)

        x = hidden[0].view(batch_size, -1)
        
        b = x.detach()
        b = F.relu(self.bn6_0(self.fc6_0(b)))
        b = self.fc6(b).squeeze(1)

        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5_0(self.fc5_0(x)))
        mu = torch.tanh(self.fc5(x))
        
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        pos = mu + noise

        # bound between [-1, 1]
        pos = torch.tanh(pos)

        log_pi = Normal(mu, self.std).log_prob(pos)
        log_pi = torch.sum(log_pi, dim=1)
         
        return hidden, mu, pos, b, log_pi

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
                
        if self.isBn:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)

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


