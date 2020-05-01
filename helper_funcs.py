from skimage.io import imread
import torch
from torchvision import transforms
import os
import re

def trans_w_seed(af, seed):
    _transform = transforms.Compose(
                                   [transforms.ToPILImage(), 
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.RandomVerticalFlip(),
#                                     transforms.RandomCrop((1536, 3072)),
                                    transforms.Resize((1536, 3072)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))
                                   ])
    if seed == 1:
        af = af[::-1, :]
    elif seed == 3:
        af = af[:, ::-1]
    af = _transform(af)
    return af
    
def read_af(input_t, batch_dir, seed):
    afs = []
    for i, subdir in enumerate(batch_dir):
        af = imread(subdir)
        af = trans_w_seed(af, i)
        afs.append(af)
    afs = torch.stack(afs).cuda()
    clean_afs = afs.clone().cuda()
    if input_t is not None:
        afs = torch.min(afs, input_t)
    return afs, clean_afs

def greedyReward(input_t, locs):
    batch_size, C, H, W = input_t.size()

    rewards = []
    
    for i in range(batch_size):
        loc = locs[i]
        window_size = 512
        x_l = int((loc[0]+1) * (H - window_size) / 2)
        y_l = int((loc[1]+1) * (W - window_size) / 2)
        x_r = int(min(H, x_l + window_size))
        y_r = int(min(W, y_l + window_size))
        if torch.mean(input_t[i][:, x_l:x_r, y_l:y_r]) > -0.5:
            r = 1
        else:
            r = 0
        rewards.append(r)
    
    rewards = torch.FloatTensor(rewards)
    if input_t.is_cuda:
        rewards = rewards.cuda()
    
    return rewards

def dist_est(n_imgs, last_dist_maps, last_move_steps):
    imgs = []
    for i in range(n_imgs.size(0)):
        n_img, last_dist_map, last_move_step = n_imgs[i], last_dist_maps[i], last_move_steps[i]
        mapa = n_img - last_move_step
        mapb = -n_img - last_move_step
        diffa = torch.abs(mapa) - last_dist_map
        diffb = torch.abs(mapb) - last_dist_map
        mask = (torch.abs(diffa) < torch.abs(diffb)).float()
        n_img = (n_img * mask + (1-mask) * (-n_img))
        imgs.append(n_img)
        
    imgs = torch.stack(imgs)
    return imgs

def dist_from_region(n_imgs, locs):
    assert len(n_imgs.shape) == 4
    
    dists = []

    batch_size, C, H, W = n_imgs.shape
    assert C == 1
    window_size =  512
    for i in range(batch_size):
        loc = locs[i]
        n_img = n_imgs[i, 0]
        x_l = int((loc[0]+1) * (H - window_size) / 2)
        y_l = int((loc[1]+1) * (W - window_size) / 2)
        x_r = int(min(H, x_l + window_size))
        y_r = int(min(W, y_l + window_size))

        dist = torch.mean(n_img[x_l:x_r, y_l:y_r])
        dists.append(dist)
        
    dists = torch.stack(dists)
    return dists

def solver(curr_pos, delta_y):

    a1 = 0.04655
    a2 = -1.9481e-5
    b = 2*a2*curr_pos+a1
    d2 = (b)**2 + 4*a2*delta_y
    sol2 = (-b + torch.sqrt(d2))/(2*a2)
    return sol2

def get_curr(batch_dir):
    curr = []
    for subdir in batch_dir:
        res = [int(i) for i in re.split('[/ .]', subdir) if i.isdigit()]
        assert len(res) == 2
        curr.append(res[1])
    curr = torch.FloatTensor(curr).cuda()
    return curr

def get_batch_dir(batch_dir, curr):
    new_dirs = []
    assert len(batch_dir) == curr.size(0)
    for i in range(len(batch_dir)):
        subdir = batch_dir[i]
        cur = curr[i]
        res = [int(i) for i in re.split('[/ .]', subdir) if i.isdigit()]
        assert len(res) == 2
        new_dir = subdir.replace(str(res[0]), str(res[0]+1)).replace(str(res[1]), str(int(cur.item())))
        if not os.path.exists(new_dir):
            new_dir = subdir.replace(str(res[1]), str(int(cur.item())))
        new_dirs.append(new_dir)
        
    return new_dirs
        