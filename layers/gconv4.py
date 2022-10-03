import torch
import numpy as np
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
from pconv import _C

__all__ = ["TopLeftPool", "TopRightPool", "BottomLeftPool", "BottomRightPool","G2Conv2d4","G2Conv2d4_random"]

class TopLeftPoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.G1_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.G1_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class TopRightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.G2_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.G2_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class BottomRightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.G4_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide = _C.G4_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class BottomLeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout = _C.G3_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout = ctx.saved_variables
        grad_input, grad_guide =_C.G3_pool_backward(input, guide, output, maxout, grad_output)
        return grad_input, grad_guide

class TopLeftPool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return TopLeftPoolFunction.apply(x, guide)

class TopRightPool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return TopRightPoolFunction.apply(x, guide)

class BottomRightPool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return BottomRightPoolFunction.apply(x, guide)

class BottomLeftPool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        guide = guide.contiguous()
        return BottomLeftPoolFunction.apply(x, guide)


import torch
import torch.nn as nn
class GConv2d4(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = [TopLeftPool(), TopRightPool(), BottomLeftPool(), BottomRightPool()]
    
    def forward(self, x, guide):
        x = torch.cat([self.pools[i](x.chunk(4, 1)[i], guide.chunk(4, 1)[i]) for i in range(4)], dim=1)
        x = super().forward(x)
        return x

import torch
class G2Conv2d4(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = GConv2d4(*args, **kwargs)
        self.conv2 = GConv2d4(*args, **kwargs)

    def forward(self, x):
        # pdb.set_trace()
        guide = x.clone()
        conv1 = self.conv1(x, guide)
        conv2 = self.conv2(conv1, guide)
        return conv2


# another feature guided by max length vector across c channels
class G2Conv2d4_len_feature(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = GConv2d4(*args, **kwargs)

    def forward(self, x):

        # x dimension [num_sampler,channel,row,column]
        # max_len_index save argmax length of each sample
        max_len_index = []
        # save different samplers' index
        sampler_index = []
        num_sampler = x.shape[0]
        channel = x.shape[1]
        for sample in range(num_sampler):
            # In each sample, we calculate the argmax coordiante in each split
            for i in range(4):

                # divide dimension -> [channel/4, row, column]
                divide = x[sample].chunk(4,0)[i]
                for c in range(int(channel/4)):
                    divide[c] = torch.mul(divide[c],divide[c])
                divide = torch.sum(divide,dim=0)
                #divide -> [row,column]
                max1,index_list1 = torch.max(divide,dim=0)
                maximum,index_list2 = torch.max(max1,dim=-1)
                coordinate = (index_list1[index_list2],index_list2)
                max_len_index.append(coordinate)
            sampler_index.append(max_len_index)
            max_len_index = []

        guide = torch.zeros_like(x)
        for sample in range(num_sampler):
            split = guide[sample].chunk(4,0)
            for i in range(4):
                for c in range(int(channel/4)):
                    coordiante_list = sampler_index[sample]
                    coordinate = coordiante_list[i]
                    split[i][c][coordinate[0]][coordinate[1]] = 1
            guide[sample] = torch.cat(split,0)
        # pdb.set_trace()
        conv1 = self.conv1(x, guide)

        return conv1

# gconv4 with fused max_length feature and max_pooling feature
# concate two feature and change channel with conv2d
class GConv2d4_len_max(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gconv_len = GConv2d4(*args, **kwargs)
        self.gconv_max = GConv2d4(*args, **kwargs)
        self.fused_conv = nn.Conv2d(args[0]*2,args[0],1)


    def forward(self, x):

        # x dimension [num_sampler,channel,row,column]
        # max_len_index save argmax length of each sample
        max_len_index = []
        # save different samplers' index
        sampler_index = []
        num_sampler = x.shape[0]
        channel = x.shape[1]
        for sample in range(num_sampler):
            # In each sample, we calculate the argmax coordiante in each split
            for i in range(4):

                # divide dimension -> [channel/4, row, column]
                divide = x[sample].chunk(4,0)[i]
                for c in range(int(channel/4)):
                    divide[c] = torch.mul(divide[c],divide[c])
                divide = torch.sum(divide,dim=0)
                #divide -> [row,column]
                max1,index_list1 = torch.max(divide,dim=0)
                maximum,index_list2 = torch.max(max1,dim=-1)
                coordinate = (index_list1[index_list2],index_list2)
                max_len_index.append(coordinate)
            sampler_index.append(max_len_index)
            max_len_index = []

        len_guide = torch.zeros_like(x)
        for sample in range(num_sampler):
            split = len_guide[sample].chunk(4,0)
            for i in range(4):
                for c in range(int(channel/4)):
                    coordiante_list = sampler_index[sample]
                    coordinate = coordiante_list[i]
                    split[i][c][coordinate[0]][coordinate[1]] = 1
            len_guide[sample] = torch.cat(split,0)
        # pdb.set_trace()
        max_guide = x.clone()
        gconv_max = self.gconv_max(x, max_guide)
        gconv_len = self.gconv_len(x,len_guide)

        fused_feature = torch.cat([gconv_len,gconv_max],dim=1)
        fused_feature = self.fused_conv(fused_feature)

        return fused_feature



class G2Conv2d4_random(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = GConv2d4(*args, **kwargs)
        self.conv2 = GConv2d4(*args, **kwargs)
        self.guide = None

    def forward(self, x):
        # guide = x.clone()
        if self.guide is None:
            self.guide = torch.rand_like(x)
        conv1 = self.conv1(x, self.guide)
        conv2 = self.conv2(conv1, self.guide)
        return conv2