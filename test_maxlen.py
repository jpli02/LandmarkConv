import torch
import numpy as np
import pdb
from torch.autograd import gradcheck
from torch.autograd import Function
import torch.nn as nn
# from layers.conv4 import TopLeftPool, TopRightPool, BottomLeftPool, BottomRightPool
from layers.maxlen import MaxLen1_Pool,MaxLen2_Pool,MaxLen3_Pool,MaxLen4_Pool

import pdb


def find_normg(input):
    # find norm guide with input:bs,channel,sh,sw
    bs = input.shape[0]
    channel = input.shape[1]
    sh = input.shape[2]
    sw = input.shape[3]
    guide = torch.zeros_like(input)
    for i in range(bs):
        for h in range(sh):
            for w in range(sw):
                for ch in range(channel):
                    guide[i][0][h][w] += guide[i][ch][h][w]*guide[i][ch][h][w]
        for c in range(channel):
            guide[i][c] = guide[i][0]
    return guide



var = torch.tensor([
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]
]).unsqueeze(0).unsqueeze(0).to(dtype=torch.double)
guide = torch.tensor([
    [4,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]
]).unsqueeze(0).unsqueeze(0).to(dtype=torch.double)

var1 = torch.tensor([
    [1,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
]).unsqueeze(0).unsqueeze(0).to(dtype=torch.double)

guide1 = torch.tensor([
    [1,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
]).unsqueeze(0).unsqueeze(0).to(dtype=torch.double)

# def maxlen1_checker(var,guide):
#     batch_size = var.shape[0]
#     channel = var.shape[1]
#     h = var.shape[2]
#     w = var.shape[3]
#     input_upper=0,input_lefter=0,input_current=0
#     for i in range(h):
#         for j in range(w):
#             input_current = var[i][j]
#             input_upper = 0, input_lefter = 0

#             if i >0:
#                 input_upper = var[i-1][j]
#                 guide_upper = guide[i-1][j]
#             if j >0:
#                 input_lefter = var[i][j-1]
#                 guide_lefter = guide[i][j-1]
            


# guide = torch.ones_like(var)
var.requires_grad = True
guide.requires_grad = True
var1.requires_grad = True
guide1.requires_grad = True



pdb.set_trace()
y = MaxLen1_Pool()(var1.cuda(), guide1.cuda())

grads={}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
y.register_hook(save_grad('y'))
var.register_hook(save_grad('x'))
guide.register_hook(save_grad('g'))
y.sum().backward()

input = torch.randn(4, 4, 8, 8,dtype=torch.double,requires_grad=True).cuda()
guide = find_normg(input).cuda()
# guide = torch.sigmoid(torch.randn(4, 4, 8, 8, dtype=torch.double,requires_grad=False)).cuda()

# pdb.set_trace()


test1 = gradcheck(lambda x, y: MaxLen1_Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
test2 = gradcheck(lambda x, y: MaxLen2_Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
test3 = gradcheck(lambda x, y: MaxLen3_Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)
test4 = gradcheck(lambda x, y: MaxLen4_Pool()(x, y), (input, guide), eps=1e-6, raise_exception=True)

print(test1)
print(test2)
print(test3)
print(test4)

