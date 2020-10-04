import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

'''
class lightUp(object):
    light up selected circular area and blur complimentary area.
    ---
    Args:
    * self._rad: radius of circle
    * self._kernel: kernel size of gaussian blur

    * x: 4D image Tensor. (batch, C, H, W)
    * l: 2D location Tensor. (batch, 2)
    ---
    Returns:
    * defocus: 4D defocused image Tensor. (batch, C, H, W)
    def __init__(self, rad, kernel):
        self._rad = rad

    def lightUpArea(self, x, l):
        pass
'''

class retina(object):

    def __init__(self, rad = 50, kernel = 25):
        self._rad = rad
        self._kernel = kernel

    def foveate(self, x, lc, isIt = True):
        ## camera model
        B, C, H, W = x.shape
        coors = (0.5 * ((lc + 1.0) * H)).long()
        defocus_imgs = []
        for i in range(B):
            im = (x[i]+1)/2
            im = np.transpose(im.numpy(), (1, 2, 0))
            im = self.looknext(im, coors[i], isIt = isIt)
            im = np.transpose(im, (-1, 0, 1))
            im = torch.from_numpy(im).unsqueeze(dim=0)
            defocus_imgs.append(im*2-1)
        defocus_imgs = torch.cat(defocus_imgs)
        return defocus_imgs
    
    def looknext(self, I, coor, isIt = True):
        ### kernel size should be odd.
        
#         print("I shape: ", I.shape)
#         print("coor shape: ", coor.shape)
        
        x_c, y_c = coor[0], coor[1]

        blurred_img = cv2.GaussianBlur(I, (self._kernel, self._kernel), 0)
        #blurred_img = np.zeros(blurred_img.shape)

        mask = np.zeros(I.shape, dtype=np.uint8)
        if isIt==False:
            self._rad  = 100
        mask = cv2.circle(mask, (x_c, y_c), self._rad, (255,255,255), -1)
        if isIt:
            out = np.where(mask==np.array([255, 255, 255]), I, blurred_img)
        else:
            out = np.where(mask!=np.array([255, 255, 255]), I, blurred_img)
        return out

class encoder(nn.Module):
    '''
    multifocus net component. Convolute an image into a feature map.
    ---
    Args:

    ---
    Returns:

    '''
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=4, padding=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        return x

class decoder(nn.Module):
    '''
    multifocus net component. decode a feature map back to an image.
    ---
    Args:

    ---
    Returns:

    '''
    def __init__(self):
        super(decoder, self).__init__()
            
        self.deconv4 = nn.ConvTranspose2d(256, 64, 4, stride=4)
        self.deconv5 = nn.ConvTranspose2d(64, 16, 4, stride=4)
        self.conv6 = nn.Conv2d(16, 1, 5, padding=2)        
  
    def forward(self, x):
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = torch.tanh(self.conv6(x))
        x = (x+1)/2
        return x
    
class mfnet(nn.Module):
    '''
    multifocus net. Fuse 2 images with diff focus position.
    ---
    Args:

    ---
    Returns:

    '''
    def __init__(self):
        super(mfnet, self).__init__()
        
        self._encoder = encoder()
        self._decoder = decoder()
        
    def forward(self, X1, X2):
        x1 = self._encoder(X1)
        x2 = self._encoder(X2)
        x = torch.cat((x1, x2), 1)
        x = self._decoder(x)
        x = x.repeat(1, 3, 1, 1)
        x = x * X1 + (1 - x) * X2
        return x

class posnet(nn.Module):
    def __init__(self):
        super(posnet, self).__init__()

        self.conv7 = nn.Conv2d(3, 16, 4, stride=4)
        self.conv8 = nn.Conv2d(16, 64, 4, stride=4)
        self.conv9 = nn.Conv2d(64, 128, 4, stride=4)

    def forward(self, x):
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))

        return x.view(x.size()[0], -1)
    
class locnet(nn.Module):
    
    def __init__(self, std = 0.10):
        super(locnet, self).__init__()
        self._posnet = posnet()
        self.fc1 = nn.Linear(4*4*128, 16)
        self.fc2 = nn.Linear(16, 2)
        self._std = std
        
    def forward(self, x):
        x = self._posnet(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        mu = torch.tanh(self.fc2(x))
        
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self._std)
        l_t = mu + noise

        # bound between [-1, 1]
        l_t = torch.tanh(l_t)
        return mu, l_t

class baselinenet(nn.Module):

    def __init__(self):
        super(baselinenet, self).__init__()
        self._posnet = posnet()
        self.fc3 = nn.Linear(4*4*128, 16)
        self.fc4 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self._posnet(x)
        x = F.leaky_relu(self.fc3(x))
        b = self.fc4(x)
        
        return b


