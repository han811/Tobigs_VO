import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.init as init
from tobi_util import ResNet, Bottleneck, ResNet_5, ResNet_5_2
from config import resnet_blk as r
 
################################
##### tobi model structure #####
################################
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)





def res_5():
    return  ResNet_5(Bottleneck, [3, 4, 6, 3])

def res_5_2():
    return  ResNet_5_2(Bottleneck, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

class Tobi_model(nn.Module):
    def __init__(self):
        super(Tobi_model, self).__init__()
        self.Res = ResNet50()
        self.Res_5 = res_5()
        self.Res_5_2 = res_5_2()
        self.ELU = nn.ELU()
        self.rnn = nn.LSTM(input_size=r['rnn1']['input_size'], hidden_size=r['rnn1']['hidden_size'], num_layers=r['rnn1']['num_layers'], batch_first=r['rnn1']['batch_first'])
        self.rnn_2 = nn.LSTM(input_size=r['rnn2']['input_size'], hidden_size=r['rnn2']['hidden_size'], num_layers=r['rnn2']['num_layers'], batch_first=r['rnn2']['batch_first'])
        self.fc1 = nn.Linear(r['fc1']['input'], r['fc1']['output']) # rcnn1
        self.fc2 = nn.Linear(r['fc2']['input'], r['fc2']['output']) # rcnn2
        self.fc3 = nn.Linear(r['fc3']['input'], r['fc3']['output']) # total
        
        #####
        self.test1 = nn.Linear(r['test1']['input'],r['test1']['output'])
        self.test2 = nn.Linear(r['test2']['input'],r['test2']['output'])
        #####
        
        # self.final_1 = nn.Linear(r['fc4']['input'], r['fc4']['output']) # translation
        # self.final_2 = nn.Linear(r['fc5']['input'], r['fc5']['output']) # quaternion
        self.final_1 = nn.Linear(1024, 3) # translation
        self.final_2 = nn.Linear(1024, 4) # quaternion

        self.norm1 = nn.BatchNorm2d(3)
        self.norm2 = nn.BatchNorm2d(3)

        for m in self.modules():
            weight_init(m)


    def forward(self, x):
        x_1 = x[1:,:]  # t-1  step
        x_2 = x[:-1,:] # t step
        # print("00",x_1.size())
        x_1 = x_1.view(5,3,224,224)
        x_2 = x_2.view(5,3,224,224)
        # print(x_1.size())
        x_1 = self.norm1(x_1)
        x_2 = self.norm2(x_2)
        # print("01",x_2.size())
        #########################
        ###Residual block pass###
        ######################### 
        x_1 = self.Res(x_1) 
        x_2 = self.Res(x_2)
        # print("02",x_2.size())

        ###########
        ###RCNN1###
        ###########
        x_3 = torch.cat([x_1,x_2],dim = 1)
        # print(x_3.size())
        x_3 = self.Res_5(x_3)
        x_3 = x_3.view(1,5,2048)
        # print(x_3.size())
        x_3, _ = self.rnn(x_3)
        x_3 = self.fc1(x_3)
        x_3 = self.ELU(x_3)

        ###########
        ###RCNN2###
        ###########
        x_2 = self.Res_5_2(x_2)
        # print("1",x_2.size())
        x_2 = x_2.view(1,5,2048)
        x_2, _ = self.rnn_2(x_2)
        # print("2",x_2.size())        
        x_2 = self.fc2(x_2)
        # print("3",x_2.size())
        x_2 = self.ELU(x_2)
        # print("4",x_2.size())

        ##############
        ###FC Layer###
        ##############
        # x_2 = x_2.view(1,r['fc1']['output'])
        # x_3 = x_3.view(1,r['fc2']['output'])
        x_3 = torch.cat([x_2,x_3],dim = 2)
        x_3 = self.fc3(x_3)
        x_3 = self.ELU(x_3)


        ####test####
        x_4 = self.test1(x_3)
        x_4 = self.ELU(x_4)
        x_5 = self.test2(x_3)
        x_5 = self.ELU(x_5)

        x_4 = self.final_1(x_4)
        x_5 = self.final_2(x_5)
        # print(x_4.size())
        # print(x_5.size())
        out = torch.cat([x_4,x_5],dim=2)
        out = out.view(5,7)
        return out

        # #Translation
        # print(x_3.size())

        # x_4 = self.final_1(x_3)

        # #Quanternion
        # x_5 = self.final_2(x_3)
        # out = torch.cat([x_4,x_5], dim = 1)
        # return out

# def forward(self, x)
def get_loss(x, y):
# def get_loss(self, x, y):
    # with torch.no_grad():
    #     out_1 = self.forward(torch.cat([x[0],x[1]],dim = 0))
    #     out_2 = self.forward(torch.cat([x[1],x[2]],dim = 0))
    #     out_3 = self.forward(torch.cat([x[2],x[3]],dim = 0))
    #     out_4 = self.forward(torch.cat([x[3],x[4]],dim = 0))
    # out_5 = self.forward(torch.cat([x[4],x[5]],dim = 0))
    
    out_1 = forward(torch.cat([x[0],x[1]],dim = 0))
    out_2 = forward(torch.cat([x[1],x[2]],dim = 0))
    out_3 = forward(torch.cat([x[2],x[3]],dim = 0))
    out_4 = forward(torch.cat([x[3],x[4]],dim = 0))
    out_5 = forward(torch.cat([x[4],x[5]],dim = 0))

    out_con = torch.cat([out_1,out_2,out_3,out_4,out_5], dim = 0)
    loss = my_loss(out_con, y)
    return loss
    # out_1 = self.forward(torch.cat([x[0],x[1]],dim = 0))
    # out_2 = self.forward(torch.cat([x[1],x[2]],dim = 0))
    # out_3 = self.forward(torch.cat([x[2],x[3]],dim = 0))
    # out_4 = self.forward(torch.cat([x[3],x[4]],dim = 0))
    # out_5 = self.forward(torch.cat([x[4],x[5]],dim = 0))

    # out_con = torch.cat([out_1,out_2,out_3,out_4,out_5], dim = 0)
    # loss = self.my_loss(out_con, y)
    # return loss

def my_loss(out,tar):
    loss = 0
    # loss += pose_loss(out[0],out[1],tar[0],tar[1])
    # loss += pose_loss(out[1],out[2],tar[1],tar[2])
    # loss += pose_loss(out[2],out[3],tar[2],tar[3])
    # loss += pose_loss(out[3],out[4],tar[3],tar[4])
    # loss += pose_loss(out[0],out[2],tar[0],tar[2])
    # loss += pose_loss(out[2],out[4],tar[2],tar[4])
    # loss += pose_loss(out[0],out[4],tar[0],tar[4]) 
    # print("now_loss0 : ",now_loss(out[0],tar[0]))
    # print("now_loss1 : ",now_loss(out[1],tar[1]))
    # print("now_loss2 : ",now_loss(out[2],tar[2]))
    # print("now_loss3 : ",now_loss(out[3],tar[3]))
    # print("now_loss4 : ",now_loss(out[4],tar[4]))
    loss += now_loss(out[0],tar[0])
    loss += now_loss(out[1],tar[1])
    loss += now_loss(out[2],tar[2])
    loss += now_loss(out[3],tar[3])
    loss += now_loss(out[4],tar[4])
    loss = loss*100
    return loss
    # loss += self.pose_loss(out[0],out[1],tar[0],tar[1])
    # loss += self.pose_loss(out[1],out[2],tar[1],tar[2])
    # loss += self.pose_loss(out[2],out[3],tar[2],tar[3])
    # loss += self.pose_loss(out[3],out[4],tar[3],tar[4])
    # loss += self.pose_loss(out[0],out[2],tar[0],tar[2])
    # loss += self.pose_loss(out[2],out[4],tar[2],tar[4])
    # loss += self.pose_loss(out[0],out[4],tar[0],tar[4]) 
    # loss += self.now_loss(out[0],tar[0])
    # loss += self.now_loss(out[1],tar[1])
    # loss += self.now_loss(out[2],tar[2])
    # loss += self.now_loss(out[3],tar[3])
    # loss += self.now_loss(out[4],tar[4])
    # return loss

def pose_loss(output_1,output_2, target_1, target_2):
    P = torch.dot(output_1, output_2)
    P_truth = torch.dot(target_1, target_2)
    loss = (P - P_truth)**2
    return loss 

def now_loss(output_1,target_1):
    return torch.mean((output_1 - target_1) ** 2)