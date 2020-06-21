
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def pose_loss(output_1,output_2, target_1, target_2):

    P = torch.dot(output_1, output_2)
    P_truth = torch.dot(target_1, target_2)
    loss = (P - P_truth)**2
    return loss 

def now_loss(output_1,target_1):
    return torch.mean((output_1 - target_1) ** 2)

def my_loss(out, tar):

    loss = 0
    loss += pose_loss(out[0],out[1],tar[0],tar[1])
    loss += pose_loss(out[1],out[2],tar[1],tar[2])
    loss += pose_loss(out[2],out[3],tar[2],tar[3])
    loss += pose_loss(out[3],out[4],tar[3],tar[4])
    loss += pose_loss(out[0],out[2],tar[0],tar[2])
    loss += pose_loss(out[2],out[4],tar[2],tar[4])
    loss += pose_loss(out[0],out[4],tar[0],tar[4]) 
    loss += now_loss(out[0],tar[0])
    loss += now_loss(out[1],tar[1])
    loss += now_loss(out[2],tar[2])
    loss += now_loss(out[3],tar[3])
    loss += now_loss(out[4],tar[4])
    return loss

class Tobi_model(nn.Module):
    def __init__(self):
        super(Tobi_model, self).__init__()
        self.Res = ResNet50()
        self.Res_5 = res_5()
        self.Res_5_2 = res_5_2()
        self.ELU = nn.ELU()
        self.rnn = nn.LSTM(input_size=int(4096), hidden_size=1000, num_layers=2, batch_first=True)
        self.rnn_2 = nn.LSTM(input_size=int(4096), hidden_size=1000, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(1000, 1024)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.final_1 = nn.Linear(1024, 3)
        self.final_2 = nn.Linear(1024, 4)

    def forward(self, x):

        x_1 = x[0,:].unsqueeze(0)  # t-1  step
        x_2 = x[1,:].unsqueeze(0) # t step
        
        #Residual block pass 
        x_1 = self.Res(x_1) 
        x_2 = self.Res(x_2)

        #RCNN1
        x_3 = torch.cat([x_1,x_2],dim = 1)
        x_3 = self.Res_5_2(x_3)
        # x_3 = nn.functional.avg_pool2d(x_3, 4)
        x_3, h_c = self.rnn(x_3)
        x_3 = self.fc1(x_3)
        #RCNN2
        x_2 = self.Res_5(x_2)
        # x_2 = nn.functional.avg_pool2d(x_2, 4)

        x_2, h_c_2 = self.rnn_2(x_2)
        x_2 = self.fc1(x_2)
        x_2 = self.ELU(x_2)
        #FC layer
        x_2 = x_2.view(1,1024)
        x_3 = x_3.view(1,1024)
        x_3 = torch.cat([x_2,x_3],dim = 1)
        # print(x_2.size())
        # print(x_3.size())
        x_3 = self.fc2(x_3)

        #Translation
        x_4 = self.fc3(x_3)
        x_4 = self.final_1(x_4)

        #Quanternion
        x_5 = self.fc3(x_3)
        x_5 = self.final_2(x_3)
        out = torch.cat([x_4,x_5], dim = 1)
        return out

    # def forward(self, x)
    def get_loss(self, x, y):
        with torch.no_grad():
            out_1 = self.forward(torch.cat([x[0],x[1]],dim = 0))
            out_2 = self.forward(torch.cat([x[1],x[2]],dim = 0))
            out_3 = self.forward(torch.cat([x[2],x[3]],dim = 0))
            out_4 = self.forward(torch.cat([x[3],x[4]],dim = 0))
        out_5 = self.forward(torch.cat([x[4],x[5]],dim = 0))
        out_con = torch.cat([out_1,out_2,out_3,out_4,out_5], dim = 0)
        loss = my_loss(out_con, y)
        return loss


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
        out = nn.functional.avg_pool2d(out, 16)
        out = out.view(1,out.size(0), -1)
        # out = self.linear(out)
        return out

class ResNet_5_2(nn.Module):
    def __init__(self, block):
        super(ResNet_5_2, self).__init__()
        self.in_planes = 2048
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
        out = nn.functional.avg_pool2d(out, 16)
        out = out.view(1,out.size(0), -1)
        # out = self.linear(out)
        return out

def res_5():
    return  ResNet_5(Bottleneck)

def res_5_2():
    return  ResNet_5_2(Bottleneck)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())


class loss_cal(nn.Module):
    def __init__(self,label):
        super(loss_cal, self).__init__()
        self.out_tensor=torch.ones(5,7)
        self.label = label
        self.num = 0

    def add_loss(self, out_):
        for i in range(5):
            if i!=4:
                with torch.no_grad():
                    self.out_tensor[i] = self.out_tensor[i+1]
            else:
                # print(out_.size())
                self.out_tensor[i] = out_
        self.num+=1
    
    def calculate_loss(self, out__):
        if self.num>=5:
            self.add_loss(out__)
            loss = my_loss(self.out_tensor,self.label[self.num-4:self.num+1,:])
            return loss
        else:
            self.add_loss(out__)
            pass



###############################
## test tensor concat sample ##
###############################
data_size = 64
batch_size = 1
input__ = Variable(torch.ones(data_size,1,3,256,192),requires_grad=False)
# input = input.int()

resnet1_4 = ResNet50()
resnet5 = res_5()

pre_out_ = 0
out_ = 0
fin_out_ = 0
im_data_stack = []
im_data_stack_2 = []

label = torch.ones(data_size,7)

loss_ = loss_cal(label)

# for i in range(int(data_size/batch_size)):
#     if i==0:
#         im_data_stack.append(input[i])
#         pre_out_ = resnet1_4.forward(im_data_stack[0])
#         pre_out_ = resnet5.forward(pre_out_)
#     elif i==1:
#         im_data_stack.append(input[i])
#         out_ = resnet1_4.forward(im_data_stack[1])
#         out_ = resnet5.forward(out_)
#         fin_out_ = torch.cat([pre_out_,out_],dim=0)
#     else:
#         pre_out_ = out_
#         del im_data_stack[0]
#         im_data_stack.append(input[i])
#         out_ = resnet1_4.forward(im_data_stack[1])
#         out_ = resnet5.forward(out_)
#         fin_out_ = torch.cat([pre_out_,out_],dim=0)
#         print(fin_out_.size()) 

# tobiVO = Tobi_model().to('cpu')
tobiVO = Tobi_model()
cal_loss = 0
#torch.save(tobiVO,'./modelsize')
# print(tobiVO.eval())
# summary(tobiVO,input_size=(3,256,192),device='cpu')
optimizer = optim.Adam(tobiVO.parameters(), lr=0.001)
for i in range(input__.size()[0] - 6):
    optimizer.zero_grad()
    new_input = input__[0+i:6+i]
    loss = tobiVO.get_loss(new_input,label[i+1:i+6])
    loss.backward(retain_graph = True)
    optimizer.step()
    print(loss)
    del loss
    del new_input
            
    