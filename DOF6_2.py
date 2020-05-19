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


class Tobi_model(nn.Module):

    def __init__(self):
        super(Tobi_model, self).__init__()
        self.Res = ResNet50()
        self.Res_5 = res_5()
        self.ELU = nn.ELU()
        self.rnn = nn.LSTM(input_size=49152, hidden_size=1000, num_layers=2, batch_first=True)
        self.rnn_2 = nn.LSTM(input_size=49152, hidden_size=1000, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(1000, 1024)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):

        x_1 = x[0]  # t-1  step
        x_2 = x[1]  # t step
        #Residual block pass & concat
        x_1 = self.Res(x_1) 
        x_2 = self.Res(x_2)
        #RCNN1
        x_3 = torch.cat([x_2,x_2],dim=0)
        x_3 = self.rnn(x_3)
        x_3 = self.fc1(x_3)
        #RCNN2
        x_2 = self.rnn_2(x_2)
        x_2 = self.fc1(x_2)
        #FC layer
        x_3 = torch.cat([x_2,x_3],dim = 0)
        x_3 = self.fc2(x_3)


        return out


## ResNet Block
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


#Bottleneck 구조
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
    #Bottle Neck 한 덩어리 이게 3,4,6,3번 반복되면 Resnet 50!
    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.elu(out)
        return out




# 5번째 ResNet 블럭
class ResNet_5(nn.Module):
    def __init__(self, block):
        super(ResNet_5, self).__init__()
        self.in_planes = 1024
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

    # 5번쨰 Resdual BLock후, 펴주자
    def forward(self, x):
        out = self.layer1_(x)
        out = self.ELU(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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
im_data_stack_2 = []

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

