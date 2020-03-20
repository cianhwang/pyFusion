import torch
import torchvision
from torchvision import transforms, datasets
import os
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

# dataset structure:

#     \root:
#         \train
#             \clean
#                 00001.png
#                 00002.png
#                 ...
#                 xxxxx.png
#             \noisy
#                 00001.png
#                 00002.png
#                 ...
#                 xxxxx.png
#         \test
#             \clean
#                 00001.png
#                 00002.png
#                 ...
#                 xxxxx.png
#             \noisy
#                 00001.png
#                 00002.png
#                 ...
#                 xxxxx.png
        

class dnDataset(torch.utils.data.Dataset):
    
    def __init__(self, _root, _isTrain = True, _transform = None):
        self.root = _root
        self.isTrain = _isTrain
        self.transform = _transform
        self.file_ids = []
        if self.isTrain:
            self.current_branch = 'train'
        else:
            self.current_branch = 'test'
            
        self.file_ids = [f for f in listdir(join(self.root, self.current_branch, 'clean')) if isfile(join(self.root, self.current_branch, 'clean', f))]
        
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):
        file_id = self.file_ids[index]

        # Load data and get label
        X = cv2.imread(join(self.root, self.current_branch, 'clean', file_id), 0)
        y = cv2.imread(join(self.root, self.current_branch, 'noisy', file_id), 0)
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
        
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y
    
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
    
#     trainset = dnDataset(_root='./1d_root/jpeg/', _isTrain=False, _transform=transforms.Compose([
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.5,), (0.5,))
#                                ]))
    
    trainset = toyDataset(10, (64, 64, 3), (2, ))

    data_loader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                              shuffle=False, num_workers=8)
    epoch = 5
    for e in range(epoch):
        for i, (X, y) in enumerate(data_loader):
            print("{}: ({}, {})".format(i, X.flatten()[0], y.flatten()[0]))