import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from config import resnet_blk as r

########################
##### ResNet Block #####
########################

class ResNet(nn.Module):
    
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = r['conv1']['out_channels']
        self.conv1 = nn.Conv2d(r['conv1']['in_channels'], r['conv1']['out_channels'], kernel_size=r['conv1']['kernel_size'],stride=r['conv1']['stride'], padding=r['conv1']['padding'], bias=r['conv1']['bias'])
        self.bn1 = nn.BatchNorm2d(r['conv1']['out_channels'])
        self.maxpool = nn.MaxPool2d(kernel_size=r['maxpool']['kernel_size'], stride=r['maxpool']['stride'], padding=r['maxpool']['padding'])
        self.layer1 = self._make_layer(block, r['conv2']['channel1'], num_blocks[0], stride=r['conv2']['stride'])
        self.layer2 = self._make_layer(block, r['conv3']['channel1'], num_blocks[1], stride=r['conv3']['stride'])
        self.layer3 = self._make_layer(block, r['conv4']['channel1'], num_blocks[2], stride=r['conv4']['stride'])
        # self.layer4 = self._make_layer(block, r['conv5']['channel1'], num_blocks[3], stride=r['conv5']['stride'])
        # self.linear = nn.Linear(r['linear']['layer1'], r['linear']['layer2'])
        self.ELU = nn.ELU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x): # 4번째 layer 및 ave_pool은 하지 않음
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.ELU(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#         out = self.layer4(out)  
#         out = nn.functional.avg_pool2d(out, 4)
        return out


##########################
##### Bottleneck 구조 #####
##########################
# bottlenet 구조는 이미 정해져 있으므로 parameter를 고정해 둔다 #

##### Bottle Neck 한 덩어리 이게 3,4,6,3번 반복되면 Resnet 50! #####

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

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_signal = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        else:
            self.shortcut_signal = False

        self.elu = nn.ELU(inplace = True) # ELU 추가됨

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut_signal == True:
            out += self.shortcut(x)
        out = self.elu(out)
        return out


###########################
##### 5번째 ResNet 블럭 #####
###########################

class ResNet_5(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_5, self).__init__()
        # self.in_planes = r['conv4']['channel3']   
        self.in_planes = r['conv4']['channel3']*2   
        self.layer = self._make_layer(block, r['conv5']['channel1'], num_blocks[2], stride=r['conv5']['stride'])
        self.ELU = nn.ELU()
        self.linear = nn.Linear(r['linear']['layer1'], r['linear']['layer2'])
       
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # 5번쨰 Resdual BLock후, 펴주자
    def forward(self, x):
        out = self.layer(x)
        out = self.ELU(out)

        ########################
        ### strange here too ###
        ########################
        # out = nn.functional.avg_pool2d(out, 16)
        out = self.pooling(out)
        # print(out.size())
        # out = out.view(1,out.size(0), -1)
        # print(out.size())
        out = out.view(1,2048, -1)

        # out = self.linear(out)
        return out


class ResNet_5_2(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_5_2, self).__init__()
        self.in_planes = r['conv4']['channel3']   
        # self.in_planes = r['conv4']['channel3']*2   
        self.layer = self._make_layer(block, r['conv5']['channel1'], num_blocks[2], stride=r['conv5']['stride'])
        self.ELU = nn.ELU()
        self.linear = nn.Linear(r['linear']['layer1'], r['linear']['layer2'])
       
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # 5번쨰 Resdual BLock후, 펴주자
    def forward(self, x):
        out = self.layer(x)
        out = self.ELU(out)

        ########################
        ### strange here too ###
        ########################
        # out = nn.functional.avg_pool2d(out, 16)
        out = self.pooling(out)
        # print(out.size())
        # out = out.view(1,out.size(0), -1)
        out = out.view(1,2048, -1)
        # print(out.size)


        # out = self.linear(out)
        return out