from toy_dataloader import toyDataset
from data_loader import load_davis_dataset
from model import focusLocNet
from toy_utils import *
import torch
from torch import optim
import torch.nn.functional as F


class Trainer(object):
    
    def __init__(self, data_loader):
        self.channel = 4
        self.hidden_size = 256
        self.batch_size = 5
        self.seq = 3
        self.model = focusLocNet(0.17, self.channel, self.hidden_size, 1).cuda()
        self.data_loader = data_loader
        self.epochs = 50
        self.lr = 1e-4
        self.optimizer= optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        
    def reset(self):
        h = [torch.zeros(1, self.batch_size, self.hidden_size).cuda(),
                      torch.zeros(1, self.batch_size, self.hidden_size).cuda()]
        l = torch.rand(self.batch_size, 1).cuda()-0.5
        return h, l
    
    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            print("train loss: {:.5f}".format(train_loss))
    
    def train_one_epoch(self, epoch):
        losses = AverageMeter()
        for i, (X, y) in enumerate(self.data_loader):
            X, y = X.cuda(), (y.cuda()-4.0)/6.0
            assert (X.max() <= 1) and (X.min() >= -1) and (y.max() <= 0.5) and (y.min() >= -0.5)
            locs = []
            locs_gt = []
            h, l = self.reset()
            for t in range(self.seq-1):
                obs = calc_obs_input(y[:, t], l)
                l_gt = calc_locs_gt(obs)
                input_t = torch.cat([X[:, t], obs], dim = 1)
                assert input_t.size() == (self.batch_size, 4, 64, 128)
                h, mu, l, b, p = self.model(input_t, l, h)
                locs.append(l)
                locs_gt.append(l_gt)
            locs = torch.stack(locs, dim = 1)
            locs_gt = torch.stack(locs_gt, dim = 1)
            
            loss = F.mse_loss(locs, locs_gt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), self.batch_size)
            

        return losses.avg

    
    
if __name__ == '__main__':
#     data_loader = torch.utils.data.DataLoader(toyDataset(100, (3, 3, 64, 128), (3, 1, 64, 128)), #loc has less size than X
#                                                        batch_size = 5,
#                                                        shuffle=True, 
#                                                        num_workers=8)
    data_loader = load_davis_dataset('../datasets/DAVIS/test_davis_video_sublist.txt', '../datasets/DAVIS/test_davis_dpt_sublist.txt', seq=3, batch_size = 5)
    Trainer(data_loader).train()