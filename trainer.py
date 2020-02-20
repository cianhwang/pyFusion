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
        
        self.std = config.std
        self.model = focusLocNet(self.std).to(self.device)
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
            self.criterion = nn.BCEWithLogitsLoss()
            self.D = init_net(NLayerDiscriminator().to(self.device))
            self.lr_gan = config.init_lr_gan
            self.optimizerD = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr_gan)

  

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
        losses_b = AverageMeter()
        losses_r = AverageMeter()
        losses_g = AverageMeter()
        losses_d = AverageMeter()
        losses = AverageMeter()
        Rs = AverageMeter()
        Ras = AverageMeter()
        Rs_model = AverageMeter()
        #Rs_random = AverageMeter()
        Rs_central = AverageMeter()
        Rs_i = AverageMeter()
        tic = time.time()
        
        with tqdm(total=self.num_train) as pbar:
            for i, (y_train, dpt) in enumerate(self.train_loader):

                y_train = y_train.to(self.device)
                dpt = dpt.to(self.device)

                self.batch_size = y_train.size()[0]
                self.seq = y_train.size()[1]
                self.model.init_hidden()

                # data shape: y_train (B, Seq, C, H, W)
                log_pi = []
                J_est = []
                #J_est_random = []
                J_est_central = []
                I_est = []
                loc = []
                l = torch.rand(self.batch_size, 2, device=self.device)*2-1
                loc.append(l)
                baselines = []

                I =  utils.getDefocuesImage(l, y_train[:, 0, ...], dpt[:, 0, ...])
                J_prev = I#y_train[:, 0, ...] ## set J_prev to be first frame of the image sequences
                #J_prev_random = I
                J_prev_central = I
                J_est.append(J_prev)
                #J_est_random.append(J_prev_random)
                J_est_central.append(J_prev_central)
                I_est.append(I)
                reward = []

                for t in range(y_train.size()[1]-1):
                    # for each time step: estimate, capture and fuse.
                    mu, l, b, p = self.model(I, l)
                    #l_random = self.RandomPolicy(l)
                    l_central = self.CentralPolicy(l)
                    log_pi.append(p)
                    I = utils.getDefocuesImage(l, y_train[:, t+1, ...], dpt[:, t+1, ...])
                    #I_random = utils.getDefocuesImage(l_random, y_train[:, t+1, ...], dpt[:, t+1, ...])
                    I_central = utils.getDefocuesImage(l_central, y_train[:, t+1, ...], dpt[:, t+1, ...])
                    J_prev = utils.fuseTwoImages(I, J_prev)
                    #J_prev_random = utils.fuseTwoImages(I_random, J_prev_random)
                    J_prev_central = utils.fuseTwoImages(I_central, J_prev_central)
                    J_est.append(J_prev)
                    #J_est_random.append(J_prev_random)
                    J_est_central.append(J_prev_central)
                    I_est.append(I)
                    loc.append(l)
                    baselines.append(b)
                    
                    r = -utils.reconsLoss(J_prev, y_train[:, t+1, ...])
                    reward.append(r)
                    for tt in range(t):
                        reward[tt] += (0.9 ** (t - tt)) * r
               

                J_est = torch.stack(J_est, dim = 1)
                #J_est_random = torch.stack(J_est_random, dim = 1)
                J_est_central = torch.stack(J_est_central, dim = 1)
                I_est = torch.stack(I_est, dim = 1)

                
                
                loc = torch.stack(loc, dim = 1)

                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                reward = torch.stack(reward).transpose(1, 0)
                R = reward
    
#                 R = -utils.reconsLoss(J_est, y_train)
#                 R = R.unsqueeze(1).repeat(1, y_train.size()[1]-1)
                R_model = -utils.reconsLoss(J_est[:, 1:], y_train[:, 1:])
                R_model = torch.mean(R_model, dim = 0)
                #R_random = -utils.reconsLoss(J_est_random[:, 1:], y_train[:, 1:])
                #R_random = torch.mean(R_random, dim = 0)
                R_central = -utils.reconsLoss(J_est_central[:, 1:], y_train[:, 1:])
                R_central = torch.mean(R_central, dim = 0)
                R_i = -utils.reconsLoss(I_est[:, 1:], y_train[:, 1:])
                R_i = torch.mean(R_i, dim = 0)
                Rs_model.update(R_model.item(),y_train.size()[0])
                #Rs_random.update(R_random.item(),y_train.size()[0])
                Rs_central.update(R_central.item(),y_train.size()[0])
                Rs_i.update(R_i.item(),y_train.size()[0])
                
                loss_baseline = F.mse_loss(baselines, R)
                
                adjusted_reward = R - baselines.detach()              

                ## Basic REINFORCE algorithm
                loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                loss = loss_baseline + loss_reinforce
                
                ## ADVERSARIAL LOSS
                ## ADDED BY QIAN, 02/05/2020
                ##------------------D------------------
                if self.use_gan:
                    self.D.zero_grad()
                    real = y_train[:, 1:].clone()
                    real = (real).view(-1, *real.shape[2:])
                    pred_labels = self.D(real)
                    real_labels = torch.ones_like(pred_labels).to(self.device)
                    errD_real = self.criterion(pred_labels, real_labels) 

                    fake = J_est[:, 1:].detach().clone()
                    fake = (fake).view(-1, *fake.shape[2:])
                    pred_labels = self.D(fake.detach())                
                    fake_labels = torch.zeros_like(pred_labels).to(self.device)
                    errD_fake = self.criterion(pred_labels, fake_labels)
                    errD = errD_real + errD_fake
                    errD.backward()
                    self.optimizerD.step()
                    ##------------------G------------------
                    pred_labels = self.D(fake)
                    real_labels = torch.ones_like(pred_labels).to(self.device)
                    loss_GAN = self.criterion(pred_labels, real_labels)

                    losses_g.update(loss_GAN.item(), y_train.size()[0])
                    losses_d.update(errD.item(), y_train.size()[0])
                    
                    loss += loss_GAN


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), y_train.size()[0])
                losses_b.update(loss_baseline.item(), y_train.size()[0])
                losses_r.update(loss_reinforce.item(), y_train.size()[0])
                

                Rs.update(torch.mean(torch.sum(R, dim = 1),dim = 0).item(),y_train.size()[0])
                Ras.update(torch.mean(torch.sum(adjusted_reward, dim = 1),dim = 0).item(),y_train.size()[0])

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
                    iteration = epoch*len(self.train_loader) + i
                    #self.writer.add_scalar('total_loss', losses.avg, iteration)
                    self.writer.add_scalars('losses', {'loss_b':losses_b.avg, 'loss_r':losses_r.avg, 'total_loss': losses.avg}, iteration)
                    #self.writer.add_scalars('rw', {'rw':Rs.avg, 'rw_a': Ras.avg}, iteration)
                    if self.use_gan:
                        self.writer.add_scalars('loss_gan', {'G':losses_g.avg, 'D': losses_d.avg}, iteration)
                    self.writer.add_scalars('rw_comp', {'model':Rs_model.avg, 'central': Rs_central.avg, 'i': Rs_i.avg}, iteration)
                    
            if self.use_tensorboard and self.is_plot:
                I_est[0] = utils.color_region(I_est[0], loc[0])
                display_tensor = torch.cat([I_est[0]/2+0.5,J_est[0]/2+0.5,y_train[0]/2+0.5], dim = 0)
                display_grid = torchvision.utils.make_grid(display_tensor, nrow = self.seq)
                self.writer.add_image('I-pred-gt', display_grid, epoch)
#                 fig = plt.figure()
# #                 if self.use_cuda:
# #                     loc = loc.cpu()
#                 plt.plot()
#                 self.writer.add_figure('reward psnr/10', fig, epoch)

            return losses.avg
    
    
#     def test(self):
        
#         self.load_checkpoint()
#         self.model.eval()
#         losses = AverageMeter()
        
#         with torch.no_grad():
#             for i, (y_test, dpt) in enumerate(self.test_loader):

#                 y_test = y_test.to(self.device)
#                 dpt = dpt.to(self.device)

#                 self.batch_size = y_test.size()[0]
#                 self.seq = y_test.size()[1]
#                 self.model.init_hidden()

#                 # data shape: y_train (B, Seq, C, H, W)
#                 log_pi = []
#                 J_est = []
#                 I_est = []
#                 loc = []
#                 l = torch.rand(self.batch_size, 2, device=self.device)*2-1
#                 loc.append(l)
#                 baselines = []

#                 I =  utils.getDefocuesImage(l, y_test[:, 0, ...], dpt[:, 0, ...])
#                 J_prev = I#y_train[:, 0, ...] ## set J_prev to be first frame of the image sequences
#                 J_est.append(J_prev)
#                 I_est.append(I)
#                 reward = []

#                 for t in range(y_train.size()[1]-1):
#                     # for each time step: estimate, capture and fuse.
#                     mu, l, b, p = self.model(I, l)
#                     log_pi.append(p)
#                     I = utils.getDefocuesImage(l, y_test[:, t+1, ...], dpt[:, t+1, ...])
#                     J_prev = utils.fuseTwoImages(I, J_prev)
#                     J_est.append(J_prev)
#                     I_est.append(I)
#                     loc.append(l)
#                     baselines.append(b)
                    
# #                     r = utils.reconsLoss(J_prev, y_test[:, t+1, ...])
# #                     reward.append(r)
# #                     for tt in range(t):
# #                         reward[tt] += 0.9 ** (t - tt) * r

#                 J_est = torch.stack(J_est, dim = 1)
#                 I_est = torch.stack(I_est, dim = 1)
#                 loc = torch.stack(loc, dim = 1)

#                 baselines = torch.stack(baselines).transpose(1, 0)
#                 log_pi = torch.stack(log_pi).transpose(1, 0)
# #                 reward = torch.stack(reward).transpose(1, 0)
# #                 R = reward# * 100
                
#                 R = -utils.reconsLoss(J_est, y_train)
#                 R = R.unsqueeze(1).repeat(1, y_train.size()[1]-1)
                
#                 loss_baseline = F.mse_loss(baselines, R)
                
#                 adjusted_reward = R - baselines.detach()              

#                 ## Basic REINFORCE algorithm
#                 loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
#                 loss_reinforce = torch.mean(loss_reinforce, dim=0)
#                 loss = loss_baseline + loss_reinforce

#                 losses.update(loss.item(), y_test.size()[0])

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

   
