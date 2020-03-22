import torch
import numpy as np

def val_from_region(X, locs):
    
    assert len(X.shape) == 3
    X_copy = X.clone()
    
    for b in range(X_copy.size(0)):
        loc = locs[b]
        
        depthmap = X_copy[b]
#         H, W = depthmap.shape
#         window_size =  min(H, W)//4
        

#         x_l = int((loc[0]+1) * (H - window_size) / 2)
#         y_l = int((loc[1]+1) * (W - window_size) / 2)
#         x_r = int(min(H, x_l + window_size))
#         y_r = int(min(W, y_l + window_size))

#         depthmap -= torch.median(depthmap[x_l:x_r, y_l:y_r])
        depthmap -= loc
    
    assert torch.all(torch.eq(X, X_copy)) == False
    return X_copy

def calc_locs_gt(obss):
    assert len(obss.size()) == 4
    ## obs -> (B, 1, H, W) cuda torch float tensor
    
    ## output -> locs(B, 1)
    batch_size = obss.size(0)
    obss_copy = obss.clone()
    locs = []
    for i in range(batch_size):
        obs = obss_copy[i]
        dist_vec = obs.flatten()
        dist_vec = dist_vec[(torch.abs(dist_vec) > 5e-2)]
        n, bins = np.histogram(dist_vec.cpu().detach().numpy())

        idx_sorted = np.argsort(n)[::-1]
        dist_to_move = (bins[idx_sorted[0]]+bins[idx_sorted[0]+1])/2
        locs.append(dist_to_move)
        obs -= dist_to_move
    
    assert torch.all(torch.eq(obss, obss_copy)) == False
    
    locs = torch.FloatTensor(locs).view(batch_size, 1)
    if obss.is_cuda:
        locs = locs.cuda()
    
    return locs

def calc_obs_input(dpts, locs):
    
    ## dpts -> (B, 1, H, W) cuda torch float tensor
    ## locs -> (B, 1)
    assert len(dpts.size()) == 4
    assert len(locs.size()) == 2
    
    ## obs -> (B, 1, H, W) cuda torch float tensor
    batch_size = dpts.size(0)
    obss = dpts.clone()
    for i in range(batch_size):
        obs = obss[i]
        loc = locs[i]
        obs -= loc
    
    assert torch.all(torch.eq(obss, dpts)) == False
    
    return obss
    
    
    

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