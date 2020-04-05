from toy_dataloader import toyDataset
from data_loader import load_davis_dataset
from model import focusLocNet
from toy_utils import *
import torch
from torch import optim
import torch.nn.functional as F


class Trainer(object):
    
    def __init__(self, data_loader):
        self.channel = 1
        self.hidden_size = 256
        self.batch_size = 16
        self.seq = 6
        self.out_size = 1
        self.model = focusLocNet(0.17, self.channel, self.hidden_size, self.out_size).cuda()
        self.data_loader = data_loader
        self.epochs = 50
        self.lr = 1e-4
        self.optimizer= optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        
    def reset(self, _batch_size):
        h = [torch.zeros(1, _batch_size, self.hidden_size).cuda(),
                      torch.zeros(1, _batch_size, self.hidden_size).cuda()]
        l = torch.rand(_batch_size, self.out_size).cuda()
        return h, l
    
    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            print("train loss: {:.5f}".format(train_loss))
    
    def train_one_epoch(self, epoch):
        losses = AverageMeter()
        for i, (X, y) in enumerate(self.data_loader):
            X, y = X.cuda(), y.cuda()
            log_pi = []
            locs = []
            baselines = []
            _batch_size = X.size(0)
            h, l = self.reset(_batch_size)
            for t in range(self.seq-1):
                h, mu, l, b, p = self.model(X[:, t], l, h)
                log_pi.append(p)
                locs.append(l)
                baselines.append(b)
                
            locs = torch.stack(locs, dim = 1)
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)
            
            R = torch.stack(reward).transpose(1, 0) * 1.0
            
            loss = F.mse_loss(locs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), self.batch_size)
            

        return losses.avg

    
    
if __name__ == '__main__':
    data_loader = torch.utils.data.DataLoader(toyDataset(100, (6, 1, 64, 128), (5, 1)), #loc has less size than X
                                                       batch_size = 16,
                                                       shuffle=True, 
                                                       num_workers=8)
    Trainer(data_loader).train()
    