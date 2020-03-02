#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import warnings
warnings.simplefilter("ignore", UserWarning)
import torchvision
import utils
from model import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from discriminator import *

# In[2]:


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

class easyAppendDict(object):
    
    def __init__(self, *argv):
        self.dict = {}
        for arg in argv:
            self.dict[arg] = []
    def __getitem__(self, key):
        if key in self.dict.keys():
            return self.dict[key]
        else: 
            raise KeyError("no such key.")
    def append(self, *argv):
        for i, (k, arg) in enumerate(zip(self.dict.keys(), argv)):
            (self.dict[k]).append(arg)
            
    def toTensor(self):
        for i, (k, v) in enumerate(self.dict.items()):
            self.dict[k] = torch.stack(v, dim = 1)
            
class prettyfloat(float):
    def __repr__(self):
        return "%0.3f" % self

# In[3]:


class Trainer(object):
    
    def __init__(self, config, data_loader):
        
        self.config = config  
        self.is_train = config.is_train
        
        if self.is_train: #
            self.train_loader = data_loader
            self.num_train = len(self.train_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
          
        self.use_cuda = config.use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.resume = config.resume
        self.model_name = 'rfc'
        self.use_tensorboard = config.use_tensorboard
        self.is_plot = config.is_plot
        
        if self.use_tensorboard:
            tensorboard_dir = config.logs_dir
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir)
        
        self.channel = config.channel
        self.hidden_size = config.hidden_size
        
        self.std = config.std
        self.model = focusLocNet(self.std, self.channel, self.hidden_size).to(self.device)
        self.RandomPolicy = RandomPolicy()
        self.CentralPolicy = CentralPolicy()
        
        self.start_epoch = 0
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.seq = config.seq
        self.lr = config.init_lr
        self.optimizer= optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        
        self.use_gan = config.use_gan
        if self.use_gan:
            self.criterion_gan = nn.BCEWithLogitsLoss()
            self.D = init_net(NLayerDiscriminator().to(self.device))
            self.lr_gan = config.init_lr_gan
            self.optimizerD = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr_gan)
            
    def reset(self):
        h = [torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                      torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)]
        l = torch.zeros(self.batch_size, 2).to(self.device)#*2-1
        return h, l
      
            
    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint()
            
        print("\n[*] Train on {} samples.".format(
            self.num_train)
        )
           
            
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss = self.train_one_epoch(epoch)

            msg1 = "train loss: {:.3f}"
            msg = msg1
            print(msg.format(train_loss))

            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 }
            )
    
    def train_one_epoch(self, epoch):
    
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = {"t_loss": AverageMeter(), "b_loss": AverageMeter(), "r_loss": AverageMeter()}
        rewards = AverageMeter()
        mselosses = AverageMeter()
        tic = time.time()
        
        with tqdm(total=self.num_train) as pbar:
            for i, (x_train, dpt) in enumerate(self.train_loader):

                x_train = x_train.to(self.device)
                dpt = dpt.to(self.device)

                self.batch_size = x_train.size(0)
                self.seq = x_train.size(1)
                
                h, l = self.reset()
                # data shape: x_train (B, Seq, C, H, W)
                data_dict = easyAppendDict("I_est", "J_est", "gaf_est", "u_est")
                loc_dict  = easyAppendDict("mus", "locs")
                log_pi = []
                baselines = []
                reward = []
                reward_wo_gamma = []

                I, gaf, u_in =  utils.getDefocuesImage(l, x_train[:, 0, ...], dpt[:, 0, ...])
                J_prev = I #x_train[:, 0, ...] ## set J_prev to be first frame of the image sequences

                data_dict.append(I, J_prev, gaf, u_in)
                loc_dict.append(l, l)
                
                if self.use_gan:
                    ## build a discriminator
                    pass

                for t in range(x_train.size(1)-1):
                    # for each time step: estimate, capture and fuse.
                    rgbd = torch.cat([I, gaf], dim = 1)
                    h, mu, l, b, p = self.model(rgbd, l, h)
                    
                    log_pi.append(p)
                    I, gaf, u_in = utils.getDefocuesImage(l, x_train[:, t+1, ...], dpt[:, t+1, ...])
                        
                    J_prev = utils.fuseTwoImages(I, J_prev)
                    data_dict.append(I, J_prev, gaf, u_in)
                    loc_dict.append(mu, l)
                    baselines.append(b)
                    
                    if self.use_gan:
                        ## treat the agent as a Generator and update rewards
                        pass
                    else:
#                         r = greedyReward(data_dict["u_est"][-2], u_in)
#                         if t == x_train.size(1)-2:
#                             data_dict.toTensor()
#                             r = r-utils.reconsLoss(data_dict["J_est"][:, 1:].detach(), x_train[:, 1:]) * 100.0
                        r = -utils.reconsLoss(J_prev.detach(), x_train[:, t+1]) * 100.0

                    reward.append(r)
                    reward_wo_gamma.append(r)
                    for tt in range(t):
                        reward[tt] = reward[tt] + (0.9 ** (t - tt)) * r
                
                data_dict.toTensor()
                loc_dict.toTensor()

                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                R = torch.stack(reward).transpose(1, 0) * 1.0
                R_wo_gamma = torch.stack(reward_wo_gamma).transpose(1, 0)

#                 R = -utils.reconsLoss(data_dict["J_est"][:, 1:].detach(), x_train[:, 1:])*100.0
#                 R = R.unsqueeze(1).repeat(1, x_train.size(1)-1)
                
                loss_baseline = F.mse_loss(baselines, R)
            
                adjusted_reward = R - baselines.detach()              

                ## Basic REINFORCE algorithm
                loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)
                
                loss = loss_reinforce + loss_baseline
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses["t_loss"].update(loss.item(), self.batch_size)
                losses["b_loss"].update(loss_baseline.item(), self.batch_size)
                losses["r_loss"].update(loss_reinforce.item(), self.batch_size)

                rewards.update(torch.mean(torch.sum(R, dim = 1),dim = 0).item(),self.batch_size)
                
                mselosses.update(torch.mean(utils.reconsLoss(data_dict["J_est"][:, 1:], x_train[:, 1:]), dim = 0).item(), self.batch_size)

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.5f}".format(
                            (toc-tic), loss.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

            if self.use_tensorboard:
                iteration = epoch#*len(self.train_loader)# + i
#                 self.writer.add_scalars('Stats/total_loss', {"t_loss":losses["t_loss"].avg, 
#                                                              "b_loss":losses["b_loss"].avg,
#                                                              "r_loss":losses["r_loss"].avg}
#                                                              , iteration)
                self.writer.add_scalar('Stats/total_loss', losses["t_loss"].avg, iteration)
                self.writer.add_scalar('Stats/reinforce_loss', losses["r_loss"].avg, iteration)
                self.writer.add_scalar('Stats/reward', rewards.avg, iteration)
                self.writer.add_scalar('Stats/mseloss', mselosses.avg*1000, iteration)

                    
            if self.use_tensorboard and self.is_plot:
                defocused = utils.color_region(data_dict["I_est"][0], loc_dict["locs"][0])
                pred = data_dict["J_est"][0]
                gt = utils.color_region(x_train[0], loc_dict["mus"][0])
#                 gaf = data_dict["gaf_est"][0].repeat(1, 3, 1, 1)
#                 gaf = utils.color_region(gaf, loc_dict["mus"][0])
                u_in = data_dict["u_est"][0].repeat(1, 3, 1, 1)
                u_in = torch.cat([u_in[:1], u_in[1:] - u_in[:-1]], dim = 0)
                u_in = utils.color_region(u_in, loc_dict["mus"][0])
                display_tensor = torch.cat([defocused, pred, gt, u_in], dim = 0)
                display_grid = torchvision.utils.make_grid(display_tensor/2+0.5, nrow = self.seq)
                self.writer.add_image('Visualization', display_grid, epoch)
                
                rw_list = map(prettyfloat, R_wo_gamma.tolist()[0])
                self.writer.add_text("reward log", '[%s]' % ', '.join(map(str, rw_list)), epoch)

            return losses["t_loss"].avg
    
    
#     def test(self):
        
#         self.load_checkpoint()
#         self.model.eval()
#         losses = AverageMeter()
        
#         with torch.no_grad():
#             for i, (x_test, dpt) in enumerate(self.test_loader):

#                 x_test = x_test.to(self.device)
#                 dpt = dpt.to(self.device)

#                 self.batch_size = x_test.size(0)
#                 self.seq = x_test.size(1)
#                 self.model.init_hidden()

#             return losses.avg
    
    def save_checkpoint(self, state):
        """
        Save a copy of the model so that it can be loaded at a future
        date.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        
        print("[*] Saved model to {}".format(self.ckpt_dir))

    def load_checkpoint(self):
        
        print("[*] Loading model from {}".format(self.ckpt_dir))
        
        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])   
        
        print("[*] Loaded model from {}".format(self.ckpt_dir))

   
