import torch
import numpy as np
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
from pconv import _C
import pdb
__all__ = ["R1Pool", "AConv2d4"]

class R1PoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout, dominant = _C.R1_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout, dominant)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout, dominant = ctx.saved_variables
        grad_input, grad_guide =_C.R1_pool_backward(input, guide, output, maxout, dominant, grad_output)
        return grad_input, grad_guide

class R2PoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout, dominant = _C.R2_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout, dominant)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout, dominant = ctx.saved_variables
        grad_input, grad_guide =_C.R2_pool_backward(input, guide, output, maxout, dominant, grad_output)
        return grad_input, grad_guide

class R3PoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout, dominant = _C.R3_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout, dominant)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout, dominant = ctx.saved_variables
        grad_input, grad_guide =_C.R3_pool_backward(input, guide, output, maxout, dominant, grad_output)
        return grad_input, grad_guide


class R4PoolFunction(Function): 
    @staticmethod
    def forward(ctx, input, guide):
        output, maxout, dominant = _C.R4_pool_forward(input, guide)
        ctx.save_for_backward(input, output, guide, maxout, dominant)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output, guide, maxout, dominant = ctx.saved_variables
        grad_input, grad_guide =_C.R4_pool_backward(input, guide, output, maxout, dominant, grad_output)
        return grad_input, grad_guide



class R1Pool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        return R1PoolFunction.apply(x, guide)

class R2Pool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        return R2PoolFunction.apply(x, guide)

class R3Pool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        return R3PoolFunction.apply(x, guide)

class R4Pool(nn.Module):
    def forward(self, x, guide):
        if guide is None:
            guide = torch.ones_like(x)
        x = x.contiguous()
        return R4PoolFunction.apply(x, guide)


import torch
import torch.nn as nn
class AConv2d4(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(AConv2d4, self).__init__(*args, **kwargs)
        self.pools = [R1Pool(), R2Pool(), R3Pool(), R4Pool()]
    
    def forward(self, x):
        x = torch.cat([self.pools[i](x.chunk(4, 1)[i]) for i in range(4)], dim=1)
        x = super().forward(x)
        return x