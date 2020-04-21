import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 
import math
import numpy as np

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generatorvanillla(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dims

        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

        self.FC = nn.Sequential(
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(self.noise_dim, self.noise_dim*2, bias=False),
            nn.BatchNorm1d(self.noise_dim*2),
            nn.LeakyReLU(0.2),
                                )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 1, 5),
            )


    def forward(self, noise, params):
        net = self.FC(noise)
        net = net.view(-1, 16, 8)
        net = self.CONV(net)    
        #net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        #net = conv1d_meta(net , self.gkernel)
        #net = torch.tanh(net* params.binary_amp) * 1.05
        net = torch.tanh((net + noise.unsqueeze(1))* 100) * 1.05
        return net

class GeneratorConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.kernelsize=3
        self.noise_dim = (params.noise_dims)
        self.dimen = int(math.sqrt(params.noise_dims))
        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=self.kernelsize),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=6,out_channels=12,kernel_size=self.kernelsize),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2),
        )
        self.FC = nn.Sequential(
            nn.Linear((self.dimen-2*self.kernelsize+2)**2*12,(self.dimen-2*self.kernelsize+2)**2*12),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear((self.dimen-2*self.kernelsize+2)**2*12, self.noise_dim*2, bias=False),
            nn.BatchNorm1d(self.noise_dim*2),
            nn.LeakyReLU(0.2),
                                )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 1, 5),
            )
     


    def forward(self, noise, params=0):
        
        noise1= noise.view(-1,1,self.dimen,self.dimen)
        net = self.conv1(noise1)
        net = self.FC(net.view(-1,12*(self.dimen-2*self.kernelsize+2)**2))
   
        #net = net.view(-1, 32, int(self.dimen/4),int(self.dimen/4))
        #net = self.CONV1(net).view(-1,1,self.noise_dim)
        
        net = net.view(-1, 16,int(self.noise_dim*2/16))
        net = self.CONV(net) 

        #net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        #net = conv1d_meta(net , self.gkernel)
        #net = torch.tanh(net* params.binary_amp) * 1.05
        net = torch.tanh((net+noise.unsqueeze(1))* 100) * 1.05
        return net

class Generator0(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dims

        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

        self.FC = nn.Sequential(
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(self.noise_dim, self.noise_dim*2, bias=False),
            nn.BatchNorm1d(self.noise_dim*2),
            nn.LeakyReLU(0.2),
                                )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 1, 5),
            )


    def forward(self, noise, params=0):
        net = self.FC(noise)
        net = net.view(-1, 16,  int(self.noise_dim/8))
        net = self.CONV(net)    
        #net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        #net = conv1d_meta(net , self.gkernel)
        #net = torch.tanh(net* params.binary_amp) * 1.05
        net = torch.tanh((net + noise.unsqueeze(1))* 100) * 1.05
        return net
        

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.kernelsize=3
        self.noise_dim = params.noise_dims
        self.dimen = int(math.sqrt(params.noise_dims))
     
        self.FC = nn.Sequential(
            #nn.Linear(self.noise_dim, self.noise_dim),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(p=0.2),
            nn.Linear(int(self.noise_dim), int(self.noise_dim), bias=False),
            nn.BatchNorm1d(int(self.noise_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
                                )
  
      
        self.ConvTran = nn.Sequential(
            ConvTranspose2d_meta(64,16, 5, stride=2,bias=False,fuc=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose2d_meta(16,4,5, stride=2,bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose2d_meta(4, 1, 5, stride=2,bias=False),
            )


    def forward(self, noise, params=0):
        
       
        
        net = self.FC(noise)
   
        net = net.view(-1, 64, int(self.dimen/8),int(self.dimen/8))
        #net = self.CONV1(net).view(-1,1,self.noise_dim)
        
        net = self.ConvTran(net)   
        net=net.reshape(-1,1,self.noise_dim)
     
        #net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        #net = conv1d_meta(net , self.gkernel)
        #net = torch.tanh(net* params.binary_amp) * 1.05
        net = torch.tanh((net+noise.unsqueeze(1))* 100) * 1.05
        return net