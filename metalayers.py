import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import scipy.stats as st
from typing import Tuple
import math
import logging

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    

def index_along(tensor, key, axis):
    indexer = [slice(None)] * len(tensor.shape)
    indexer[axis] = key
    return tensor[tuple(indexer)]


def pad_periodic(inputs, padding: int, axis: int, center: bool = True):
    
    if padding == 0:
        return inputs
    if center:
        if padding % 2 != 0:
            raise ValueError('cannot do centered padding if padding is not even')
        inputs_list = [index_along(inputs, slice(-padding//2, None), axis),
                       inputs,
                       index_along(inputs, slice(None, padding//2), axis)]
    else:
        inputs_list = [inputs, index_along(inputs, slice(None, padding), axis)]
    return torch.cat(inputs_list, dim=axis)


def pad1d_meta(inputs, padding: int):
    return pad_periodic(inputs, padding, axis=-1, center=True)


def gkern1D(kernlen=7, nsig=4):
    """Returns a 1D Gaussian kernel array."""

    x_cord = torch.arange(0., kernlen)

    mean = (kernlen - 1)/2.
    variance = nsig**2.

    # variables (in this case called x and y)
    gaussian_kernel = 1./(2.*math.pi*variance)**0.5 * torch.exp(-(x_cord - mean)**2. / (2.*variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel.type(Tensor).requires_grad_(False)



def conv1d(inputs, kernel, padding='same'):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 1d kernel
    """
    B, C, _ = inputs.size()
    kH = kernel.size()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, C, 1)

    if padding == 'valid':
        return F.conv1d(inputs, kernel)
    elif padding == 'same':
        pad = (kH-1)//2
        return F.conv1d(inputs, kernel, padding = pad)


def conv1d_meta(inputs, kernel):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 1d kernel
    """
    kH = kernel.size(0)
    padded_inputs = pad1d_meta(inputs, kH-1)
    
    return conv1d(padded_inputs, kernel, padding='valid')


class AvgPool1d_meta(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        self.padding = kernel_size - 1
        self.avgpool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        return self.avgpool1d(padded_inputs) 


class ConvTranspose1d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1):
        super().__init__()
        self.padding = kernel_size - 1 #=4
        self.trim = self.padding * stride // 2 #4
        pad = (kernel_size - stride) // 2 #=1
        self.output_padding = (kernel_size - stride) % 2 #=1
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=pad,
                                          output_padding=0, groups=groups, bias=bias, dilation=dilation)
    
    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        #print("padded_inputs:",padded_inputs.size())
        padded_outputs = self.conv1d_transpose(padded_inputs)
        #print("padded_outputs:",padded_outputs.size())
        if self.output_padding:
            padded_outputs = padded_outputs[:, :, 1:]
            #print("padded_outputs:",padded_outputs.size())

        if self.trim:
            #print("padded_outputs1:",padded_outputs[:, :, self.trim:-self.trim].size())
            return padded_outputs[:, :, self.trim:-self.trim]
            
        else:
            return padded_outputs
 
    
class Conv1d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1)*dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = 0, 
                                dilation = dilation, groups = groups, bias = bias)
    
    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        outputs = self.conv1d(padded_inputs)
        return outputs

class ConvTranspose2d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1):
        super().__init__()
        self.conv2d=nn.ConvTranspose2d(in_channels, out_channels, 2, stride, padding=0,
                                          output_padding=0, groups=1, bias=bias, dilation=dilation)
    
    def forward(self, inputs):
        return self.conv2d(inputs)


class ConvTranspose2d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1,fuc=0):
        super().__init__()
        self.fuc=fuc
        self.padding = kernel_size - 1 #=4
        self.trim = self.padding * stride // 2 #4
        pad = (kernel_size - stride) // 2 #=1
        self.output_padding = (kernel_size - stride) % 2 #=1
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=pad,
                                          output_padding=0, groups=groups, bias=bias, dilation=dilation)
    
    def forward(self, inputs):
        #print("inputs:",inputs.size())
        #print("inputs1:", self.padding)
    
        padded_inputs = pad2d_meta(inputs, self.padding,self.fuc)
        #print("padded_inputs after pad:",padded_inputs.size())
        padded_outputs = self.conv2d_transpose(padded_inputs)
        #print("padded_outputs after conv:",padded_outputs.size())
        if self.output_padding:
            padded_outputs = padded_outputs[:, :, 1:,1:]
            #print("padded_outputs:",padded_outputs.size())

        if self.trim:
            #print("padded_outputs after trim:",padded_outputs[:, :, self.trim:-self.trim,self.trim:-self.trim].size())
            return padded_outputs[:, :, self.trim:-self.trim,self.trim:-self.trim]
            
        else:
            return padded_outputs

def pad2d_meta(inputs, padding: int,fuc=0):
    return pad_periodic_2d(inputs, padding, axis=-1, center=True,fuc1=fuc)

def pad_periodic_2d(inputs, padding: int, axis: int, center: bool = True,fuc1=0):
    
    #inp=inputs.cpu().detach()
   # dim=inp.size(3)
    if padding == 0:
        return inputs
    if fuc1==1:  
       # print("dimtype:",type(dim))
        padpic2=inputs.repeat(1,1,1+padding,1+padding)
        return padpic2  
    if center:
        if padding % 2 != 0:
            raise ValueError('cannot do centered padding if padding is not even')
        inputs_list = [index_along(inputs, slice(-padding//2, None), axis),
                       inputs,
                       index_along(inputs, slice(None, padding//2), axis)]
    else:
        inputs_list = [inputs, index_along(inputs, slice(None, padding), axis)]
    
    padpic1=torch.cat(inputs_list, dim=axis)
    if center:
        if padding % 2 != 0:
            raise ValueError('cannot do centered padding if padding is not even')
        inputs_list = [index_along(padpic1, slice(-padding//2, None), axis-1),
                       padpic1,
                       index_along(padpic1, slice(None, padding//2), axis-1)]
    else:
        inputs_list = [padpic1, index_along(padpic1, slice(None, padding), axis-1)]
    
    padpic2=torch.cat(inputs_list, dim=axis-1)

    return padpic2
       