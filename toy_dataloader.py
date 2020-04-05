import torch
import torchvision
from torchvision import transforms, datasets
import os
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
    
class toyDataset(torch.utils.data.Dataset):
    
    def __init__(self, _length, _X_size, _y_size, _transform = None):
        self.length = _length
        self.X_size = _X_size
        self.y_size = _y_size
        self.transform = _transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Load data and get label
        torch.manual_seed(index)
        X = torch.rand(self.X_size)*2.0-1.0
        y = torch.rand(self.y_size)*2.0-1.0
        
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y

if __name__ == '__main__':
    
    trainset = toyDataset(10, (64, 64, 3), (2, ))

    data_loader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                              shuffle=False, num_workers=8)
    epoch = 5
    for e in range(epoch):
        for i, (X, y) in enumerate(data_loader):
            print("{}: ({}, {})".format(i, X.flatten()[0], y.flatten()[0]))