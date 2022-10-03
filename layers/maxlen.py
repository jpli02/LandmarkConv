import torch
import numpy as np
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
from pconv import _C
from ..layers.conv4 import PConv2d4

__all__ = [ "MaxLen1_Pool","MaxLen2_Pool","MaxLen3_Pool","MaxLen4_Pool",
            "MaxLenConv2d4","FusedConv2d4"]

##########################################################################
# Implementation for feature guided by max length across every channels
# This feature will be fused with max pooling feature
##########################################################################
class MaxLen1_PoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.maxlen1_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        # pdb.set_trace()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.maxlen1_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class MaxLen2_PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.maxlen2_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.maxlen2_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class MaxLen3_PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.maxlen3_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide = _C.maxlen3_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class MaxLen4_PoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.maxlen4_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.maxlen4_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class MaxLen1_Pool(nn.Module):
    def forward(self, x, guide):
        # pdb.set_trace()
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return MaxLen1_PoolFunction.apply(x, guide)

class MaxLen2_Pool(nn.Module):
    def forward(self, x, guide):
        # pdb.set_trace()
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return MaxLen2_PoolFunction.apply(x, guide)

class MaxLen3_Pool(nn.Module):
    def forward(self, x, guide):
        # pdb.set_trace()
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()

        return MaxLen3_PoolFunction.apply(x, guide)

class MaxLen4_Pool(nn.Module):
    def forward(self, x, guide):
        # pdb.set_trace()
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return MaxLen4_PoolFunction.apply(x, guide)



# another feature guided by max length vector across c channels
class MaxLenConv2d4(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaxLenConv2d4,self).__init__(*args, **kwargs)
        self.pools = [MaxLen1_Pool(), MaxLen2_Pool(), MaxLen3_Pool(), MaxLen4_Pool()]
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # pdb.set_trace()
        num_sampler = x.shape[0]
        channel = x.shape[1]
        width = x.shape[2]
        height = x.shape[3]
        # norm guide scaled by min_norm and softmax to avoid NAN
        guide = torch.norm(x,dim=1)
        guide = guide.view(num_sampler,width*height)
        min_norm = torch.min(guide,dim=1)[0].view(num_sampler,1).repeat(1,width*height)
        guide = guide.div(channel)
        guide = self.softmax(guide).view(num_sampler,1,width,height)
        guide = guide.repeat(1,channel,1,1)

        # guide.shape should be the same as x.shape
        x = torch.cat([self.pools[i](x.chunk(4, 1)[i], guide.chunk(4, 1)[i]) for i in range(4)], dim=1)
        x = super().forward(x)
        return x


# Fused feature for max_pooling and max_len 
# concate two feature and change channel with conv2d

class FusedConv2d4(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(FusedConv2d4,self).__init__(*args, **kwargs)
        # pdb.set_trace()
        self.NormConv = MaxLenConv2d4(*args, **kwargs)
        self.MaxpoolConv = PConv2d4(*args, **kwargs)
        self.channelConv = nn.Conv2d(args[0]+args[0],args[1],1)


    def forward(self, x):

        indentity = x
        Norm_feature = self.NormConv(indentity)
        Maxpool_feature = self.MaxpoolConv(indentity)
        # input_channel = indentity.shape[1]
        # Norm_channel = Norm_feature.shape[1]
        # Maxpool_channel = Maxpool_feature.shape[1]
        x = torch.cat([Norm_feature,Maxpool_feature],dim=1)
        x = self.channelConv(x)

        x = super().forward(x)

        return x


        