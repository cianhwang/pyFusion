#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import os
import time
import warnings
warnings.simplefilter("ignore", UserWarning)
import torchvision
import utils
from model import focusLocNet
from tensorboardX import SummaryWriter
from tqdm import tqdm


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
        self.resume = config.resume
        self.model_name = config.model_name
        self.use_tensorboard = config.use_tensorboard
        self.is_plot = config.is_plot
        
        if self.use_tensorboard:
            tensorboard_dir = 'runs/robotics_focus_control'#self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir)
        
        self.std = config.std
        self.model = focusLocNet(self.std).to(self.device)
        
        self.start_epoch = 0
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.lr = config.init_lr
        self.optimizer= optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
  

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
        losses = AverageMeter()
        tic = time.time()
        
        with tqdm(total=self.num_train) as pbar:
            for i, (y_train, dpt) in enumerate(self.train_loader):

                y_train = y_train.to(self.device)
                dpt = dpt.to(self.device)

                self.batch_size = y_train.size()[0]
                seq = y_train.size()[1]
                self.model.init_hidden()

                # data shape: y_train (B, Seq, C, H, W)
                log_pi = []
                J_est = []
                J_prev = y_train[:, 0, ...] ## set J_prev to be first frame of the image sequences
                J_est.append(J_prev)

                for t in range(y_train.size()[1]-1):
                    # for each time step: estimate, capture and fuse.
                    mu, l, p = self.model(J_prev)
                    log_pi.append(p)
                    I = utils.getDefocuesImage(l, y_train[:, t+1, ...], dpt[:, t+1, ...])
                    J_prev = utils.fuseTwoImages(I, J_prev)
                    J_est.append(J_prev)

                J_est = torch.stack(J_est, dim = 1)

                log_pi = torch.stack(log_pi).transpose(1, 0)
                R = -utils.reconsLoss(J_est, y_train)
                R = R.unsqueeze(1).repeat(1, y_train.size()[1]-1)

                ## Basic REINFORCE algorithm
                loss = torch.sum(-log_pi*R, dim=1)
                loss = torch.mean(loss, dim=0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), y_train.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f}".format(
                            (toc-tic), loss.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + i
                    self.writer.add_scalar('train_loss', losses.avg, iteration)
                    
                    if self.is_plot:
                        display_tensor = torch.cat([J_est[0],y_train[0]], dim = 0)
                        display_grid = torchvision.utils.make_grid(display_tensor, nrow = seq)
                        self.writer.add_image('pred-gt', display_grid, iteration)

            return losses.avg
    
    
    def test(self):
        
        self.load_checkpoint()
        self.model.eval()
        losses = AverageMeter()
        
        with torch.no_grad():
            for i, (y_test, dpt) in enumerate(self.test_loader):

                y_test = y_test.to(self.device)
                dpt = dpt.to(self.device)

                self.batch_size = y_test.size()[0]
                self.model.init_hidden()

                # data shape: y_train (B, Seq, C, H, W)
                log_pi = []
                J_est = []
                J_prev = y_test[:, 0, ...] ## set J_prev to be first frame of the image sequences
                J_est.append(J_prev)

                for t in range(y_test.size()[1]-1):
                    # for each time step: estimate, capture and fuse.
                    mu, l, p = self.model(J_prev)
                    log_pi.append(p)
                    I = utils.getDefocuesImage(l, y_test[:, t+1, ...], dpt[:, t+1, ...])
                    J_prev = utils.fuseTwoImages(I, J_prev)
                    J_est.append(J_prev)

                J_est = torch.stack(J_est, dim = 1)

                log_pi = torch.stack(log_pi).transpose(1, 0)
                R = -utils.reconsLoss(J_est, y_test)
                R = R.unsqueeze(1).repeat(1, y_test.size()[1]-1)

                ## Basic REINFORCE algorithm
                loss = torch.sum(-log_pi*R, dim=1)
                loss = torch.mean(loss, dim=0)

                losses.update(loss.item(), y_test.size()[0])

            return losses.avg
    
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
   