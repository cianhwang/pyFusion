import torch
import torchvision
from torchvision import transforms, datasets
import os
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage.io import imread
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

def gci(filepath):
    files = os.listdir(filepath)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    if '.ipynb_checkpoints' in files:
        files.remove('.ipynb_checkpoints')
    return sorted(files)

def get_jpg_files(path):

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
                
    return files
    
class afDataset(torch.utils.data.Dataset):
    
    def __init__(self, _path, _transform = transforms.Compose(
                               [transforms.ToPILImage(), 
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomCrop((1536, 3072)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))
                               ])):
        self.path = _path
        self.transform = _transform
        self.files = gci(self.path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        X = imread(os.path.join(self.path, self.files[index]))
        
        if self.transform:
            X = self.transform(X)

        return X
    
class afPathDataset(torch.utils.data.Dataset):
    
    def __init__(self, _path):
        self.path = _path
        self.files = get_jpg_files(self.path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        return self.files[index]
    
class afDataLoader():
    def __init__(self, _path, _batch_size, valid_size = 0.2, shuffle = True):
            
        dataset = afPathDataset(_path)
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        if shuffle:
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
            
        self.train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False,
                                                       num_workers=int(8))
        self.valid_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       sampler=valid_sampler,
                                                       shuffle=False,
                                                       num_workers=int(8))
        self.dataset = dataset

    def load_data(self):
        return (self.train_loader, self.valid_loader)

    def __len__(self):

        return len(self.dataset)
    
def load_af_dataset(af_path, batch_size = 5):
    video_data_loader = afDataLoader(af_path, batch_size)
    video_dataset = video_data_loader.load_data()
    return video_dataset

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(20,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    
    trainset, validset = load_af_dataset('/home/qian/Documents/datasets/afs')
    dataiter = iter(trainset)
    x_train = dataiter.next()
    imshow(torchvision.utils.make_grid(x_train))