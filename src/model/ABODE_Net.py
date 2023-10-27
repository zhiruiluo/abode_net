import logging
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils import spectral_norm


logger = logging.getLogger(__name__)


class FCN_Block2d(nn.Module):
    def __init__(self, inplane, outplane, kernel, stride, padding, bias, batchnorm) -> None:
        super().__init__()
        layers = []
        layers += [nn.Conv2d(inplane, outplane, kernel, stride, padding, bias=bias)]
        if batchnorm:
            layers += [nn.BatchNorm2d(outplane)]
        layers += [nn.ReLU(inplace=True)]
        self.fcnblock = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.fcnblock(x)
        return x


class FCN_Base(nn.Module):
    def __init__(self, in_channels, chan_list, ker_list, padding_list, stride_list, bn=True):
        super().__init__()
        self.chan_list = chan_list
        self.ker_list = ker_list
        self.padding_list = padding_list
        self.stride_list = stride_list
        layers = []
        for i in range(len(chan_list)):
            if i == 0:
                inplane = in_channels
            else:
                inplane = chan_list[i-1]
            
            layers += [FCN_Block2d(inplane=inplane,outplane=chan_list[i],kernel=ker_list[i],stride=stride_list[i],padding=padding_list[i],bias=True,batchnorm=bn)]
        self.num_layers = len(layers)
        self.model = nn.Sequential(*layers)

        self.out_channel = chan_list[-1]

    def conv_out_shape(self, in_size, padding, stride, kernel, dilation):
        out_size = math.floor((in_size + 2 * padding - dilation * (kernel - 1) - 1)/ stride + 1)
        return out_size

    def fcnzl_out(self, var_len, time_len):
        for i in range(self.num_layers):
            var_len = self.conv_out_shape(var_len, self.padding_list[i][0], self.stride_list[i][0], self.ker_list[i][0], 1)
            time_len = self.conv_out_shape(time_len, self.padding_list[i][1], self.stride_list[i][1], self.ker_list[i][1], 1)

        return var_len, time_len

    def forward(self, x):
        x = self.model(x)
        return x

class FCNZL2D(FCN_Base):
    chan_list = [128, 256, 128]
    ker_list = [(8,8),(5,5),(3,3)]
    padding_list = [(3,3),(2,2),(1,1)]
    stride_list = [(4,4),(2,2),(1,1)]
    def __init__(self, in_channels, bn=True):
        super().__init__(in_channels, self.chan_list, self.ker_list, self.padding_list, self.stride_list, bn)

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, mul_input = True):
        super(SELayer, self).__init__()
        self.mul_input = mul_input
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # logger.debug(f'SElayer {y.shape} y {y.expand_as(x).shape}')
        if self.mul_input:
            return x * y.expand_as(x)
        else:
            return y.expand_as(x)

class vAttn(nn.Module):
    def __init__(self, in_channels, add_input=True):
        super().__init__()
        out_channels = in_channels//8
        self.add_input = add_input
        self.query_w = conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_v = self.query_w(x) # (B,C,V,T)
        query_v = rearrange(query_v, 'b c v t -> (b t) v c')

        key_v = self.key_w(x)     # (B,C,V,T)
        key_v = rearrange(key_v, 'b c v t -> (b t) c v')
        
        value_v = self.value_w(x) # (B,C1,V,T)
        value_v = rearrange(value_v, 'b c1 v t -> (b t) v c1')

        query_v = torch.tanh(query_v)
        key_v = torch.tanh(key_v)
        # softmax at (B T V V)
        attn = self.softmax(torch.bmm(query_v, key_v))
        
        attn_value = rearrange(torch.bmm(attn, value_v), '(b t) v c1 -> b c1 v t', b=B)
        attn_value = self.attn_value_w(attn_value)  # (B,C1,V,T) -> (B,C,V,T)

        if self.add_input:
            return x + self.sigma * attn_value
        else:
            return self.sigma * attn_value


class tAttn(nn.Module):
    def __init__(self, in_channels, add_input=True):
        super().__init__()
        out_channels = in_channels//8
        self.add_input = add_input
        self.query_w =conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_v = self.query_w(x)  # (B,C,V,T)
        query_v = rearrange(query_v, 'b c v t -> (b v) t c')

        key_v = self.key_w(x)  # (B,C,V,T)
        key_v = rearrange(key_v, 'b c v t -> (b v) c t')

        value_v = self.value_w(x) # (B,C1,V,T)
        value_v = rearrange(value_v, 'b c1 v t -> (b v) t c1')

        query_v = torch.tanh(query_v)
        key_v = torch.tanh(key_v)
        # attention softmax at (B V T T)
        attn = self.softmax(torch.bmm(query_v, key_v))

        attn_value = rearrange(torch.bmm(attn, value_v), '(b v) t c1 -> b c1 v t',b=B)
        attn_value = self.attn_value_w(attn_value) # (B C1 V T) (B,C,V,T)

        if self.add_input:
            return x + self.sigma * attn_value #, attn.view(B,V,T,T)
        else:
            return self.sigma * attn_value


class ParallelAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_dim = in_dim
        self.vattn = vAttn(in_dim,add_input=False)
        self.tattn = tAttn(in_dim, add_input=False)
        self.selayer = SELayer(in_dim,mul_input=False)
        
    def forward(self, x):
        x1 = self.vattn(x)
        x2 = self.tattn(x)
        x3 = self.selayer(x)
        x = (x + x1 + x2) * x3
        return x


class ABODE_Net(nn.Module):
    def __init__(self, nclass) -> None:
        super().__init__()
        
        self.fcn = FCNZL2D(1, True)
        self.atten = ParallelAttention(self.fcn.out_channel)
        
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        in_classifier = self.fcn.out_channel
        
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.4),
            spectral_norm(nn.Linear(in_classifier, nclass))
        )
        
    def forward(self, x):
        x = self.fcn(x)
        x = self.atten(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def test_abode_net():
    model = ABODE_Net(nclass=2)
    model(torch.random)