from toy_dataloader import toyDataset
from model import focusLocNet
import torch
from torch import optim
import torch.nn.functional as F

def val_from_region(X, locs):
    
    assert len(X.shape) == 3
    X_copy = X.clone()
    
    for b in range(X_copy.size(0)):
        loc = locs[b]
        
        depthmap = X_copy[b]
        H, W = depthmap.shape
        window_size =  min(H, W)//4

        x_l = int((loc[0]+1) * (H - window_size) / 2)
        y_l = int((loc[1]+1) * (W - window_size) / 2)
        x_r = int(min(H, x_l + window_size))
        y_r = int(min(W, y_l + window_size))

        depthmap -= torch.median(depthmap[x_l:x_r, y_l:y_r])
    
    assert torch.all(torch.eq(X, X_copy)) == False
    return X_copy

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    
    def __init__(self, data_loader):
        self.channel = 4
        self.hidden_size = 256
        self.batch_size = 5
        self.seq = 3
        self.model = focusLocNet(0.17, self.channel, self.hidden_size).cuda()
        self.data_loader = data_loader
        self.epochs = 50
        self.lr = 1e-4
        self.optimizer= optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        
    def reset(self):
        h = [torch.zeros(1, self.batch_size, self.hidden_size).cuda(),
                      torch.zeros(1, self.batch_size, self.hidden_size).cuda()]
        l = torch.zeros(self.batch_size, 2).cuda()
        return h, l
    
    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            print("train loss: {:.5f}".format(train_loss))
    
    def train_one_epoch(self, epoch):
        losses = AverageMeter()
        for i, (X, y) in enumerate(self.data_loader):
            X, y = X.cuda()/2.0, y.cuda()
            locs = []
            h, l = self.reset()
            for t in range(self.seq-1):
                input_t = val_from_region(X[:, -1], l).unsqueeze(1)
                input_t = torch.cat([X[:, :-1], input_t], dim = 1)
                assert input_t.size() == (self.batch_size, 4, 64, 128)
                h, mu, l, b, p = self.model(input_t, l, h)
                locs.append(l)
            locs = torch.stack(locs, dim = 1)
            
            loss = F.mse_loss(locs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), self.batch_size)
            

        return losses.avg

    
    
if __name__ == '__main__':
    data_loader = torch.utils.data.DataLoader(toyDataset(100, (4, 64, 128), (2, 2)), #loc has less size than X
                                                       batch_size = 5,
                                                       shuffle=True, 
                                                       num_workers=8)
    Trainer(data_loader).train()