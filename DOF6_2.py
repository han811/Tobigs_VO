import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from params import par
from torch.nn.init import kaiming_normal_, orthogonal_

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048, 1024)
        self.ELU = nn.ELU()
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ELU(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#         out = self.layer4(out)
#         out = nn.functional.avg_pool2d(out, 4)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.elu = nn.ELU(inplace = True) # ELU 추가됨

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.elu(out)
        return out




class ResNet_5(nn.Module):
    def __init__(self, block):
        super(ResNet_5, self).__init__()
        
        self.in_planes = 1024
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)        
        self.layer1_ = self._make_layer(block, 512, 3, stride=2)
        self.ELU = nn.ELU()
        self.linear = nn.Linear(2048, 1024)
        
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1_(x)
        print("please1")
        out = self.ELU(out)
        
        
        print("please2")
        
        out = nn.functional.avg_pool2d(out, 4)
        
        print("please3")
#         out = out.view(out.size(0), -1)
        print("please4")
        
#         out = self.linear(out)
        return out



def res_5():
    return  ResNet_5(Bottleneck)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())





###############################
## test tensor concat sample ##
###############################
data_size = 64
batch_size = 1
input = torch.rand(data_size,1,3,256,192)

resnet1_4 = ResNet50()
resnet5 = res_5()

pre_out_ = 0
out_ = 0
fin_out_ = 0
im_data_stack = []

for i in range(int(data_size/batch_size)):
    if i==0:
        im_data_stack.append(input[i])
        pre_out_ = resnet1_4.forward(im_data_stack[0])
        pre_out_ = resnet5.forward(pre_out_)
    elif i==1:
        im_data_stack.append(input[i])
        out_ = resnet1_4.forward(im_data_stack[1])
        out_ = resnet5.forward(out_)
        fin_out_ = torch.cat([pre_out_,out_],dim=0)
    else:
        pre_out_ = out_
        del im_data_stack[0]
        im_data_stack.append(input[i])
        out_ = resnet1_4.forward(im_data_stack[1])
        out_ = resnet5.forward(out_)
        fin_out_ = torch.cat([pre_out_,out_],dim=0)
        print(fin_out_.size()) 

