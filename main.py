import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tobi_util import ResNet, Bottleneck, ResNet_5 , ResNet_5_2
from config import resnet_blk as r
from model import Tobi_model, my_loss, get_loss

import os
from params import par
from data_helper import get_data_info_tobi, ImageSequenceDataset

from torch.utils.tensorboard import SummaryWriter


train=True
gpu=False
predict_path = os.getcwd()+'/result/'
model_path = os.getcwd()+'/weight'
log_path = os.getcwd()+'/log'

writer = SummaryWriter(log_path)

test_num=10


# should fix for when batch != 1
def run(model,data,optimizer,epoch,scheduler,batch=1):
    best_val = 10**8
    n = len(data)//batch
    # r = len(data)%batch
    loss = 0
    for j in range(epoch):
        sum_loss = 0
        for i in range(n):
            if gpu==False:
                input_ = torch.FloatTensor(data[i][0])
                input_.requires_grad=True
                input_ = torch.reshape(input_,(6,1,3,224,224))
                label_ = torch.FloatTensor(data[i][1])



                # if batch==1:  
                #     input_ = torch.FloatTensor(data[i][0])
                #     input_.requires_grad=True
                #     input_ = torch.reshape(input_,(6,1,3,224,224))
                #     label_ = torch.FloatTensor(data[i][1])
                # else:
                #     for k in range(batch):
                #         if k==0:
                #             input_ = torch.FloatTensor(data[i*batch+k][0])
                #             input_.requires_grad=True
                #             input_ = torch.reshape(input_,(6,1,3,224,224))
                #             label_ = torch.FloatTensor(data[i*batch+k][1])
                #         else:
                #             input_temp = torch.FloatTensor(data[i*batch+k][0])
                #             input_temp.requires_grad=True
                #             input_temp = torch.reshape(input_temp,(6,1,3,224,224))
                #             input_ = torch.cat([input_,input_temp],dim=0)
                #             label_temp = torch.FloatTensor(data[i*batch+k][1])
                #             label_ = torch.cat([label_,label_temp],dim=0)
                    # input_ = torch.reshape(input_,(6,batch,3,224,224)) # how to set batch training???
                    # print(input_.size())
            else:
                input_ = torch.FloatTensor(data[i][0])
                input_.requires_grad=True
                input_ = torch.reshape(input_,(6,1,3,224,224)).cuda()
                label_ = torch.FloatTensor(data[i][1]).cuda()





                # if batch==1:
                #     input_ = torch.FloatTensor(data[i][0])
                #     input_.requires_grad=True
                #     input_ = torch.reshape(input_,(6,1,3,224,224)).cuda()
                #     label_ = torch.FloatTensor(data[i][1]).cuda()
                # else:
                #     for k in range(batch):
                #         if k==0:
                #             input_ = torch.FloatTensor(data[i*batch+k][0])
                #             input_.requires_grad=True
                #             input_ = torch.reshape(input_,(6,1,3,224,224)).cuda()
                #             label_ = torch.FloatTensor(data[i*batch+k][1]).cuda()
                #         else:
                #             input_temp = torch.FloatTensor(data[i*batch+k][0])
                #             input_temp.requires_grad=True
                #             input_temp = torch.reshape(input_temp,(6,1,3,224,224)).cuda()
                #             input_ = torch.cat([input_,input_temp],dim=0).cuda()
                #             label_temp = torch.FloatTensor(data[i*batch+k][1]).cuda()
                #             label_ = torch.cat([label_,label_temp],dim=0).cuda()
            # loss = model.get_loss(input_,label_[1:6])
            out_1 = model(torch.cat([input_[0],input_[1]],dim=0))
            out_2 = model(torch.cat([input_[1],input_[2]],dim=0))
            out_3 = model(torch.cat([input_[2],input_[3]],dim=0))
            out_4 = model(torch.cat([input_[3],input_[4]],dim=0))
            out_5 = model(torch.cat([input_[4],input_[5]],dim=0))
            out_con = torch.cat([out_1,out_2,out_3,out_4,out_5],dim=0)
            
            loss = loss + my_loss(out_con, label_[1:6])
            # print(loss)
            if((i>=batch-1) and ((i+1)%batch==0)):
                optimizer.zero_grad()
                sum_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                print('{} / {} || epoch : {} || loss : {}'.format((i+1)//batch, n, j+1, loss))
                loss = 0
            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)

            # optimizer.step()
            # print('{} / {} || epoch : {} || loss : {}'.format(i, n, j, loss), end='\r')
            # print(loss)
        if(best_val>sum_loss):
            writer.add_scalar('train_loss',sum_loss,j)
            best_val = sum_loss
        sum_loss = 0
        torch.save(tobiVO.state_dict(),model_path+'/modelsize'+str(j)+'.pth')

if __name__ == "__main__":
    train_df = get_data_info_tobi(folder_list=par.train_video, overlap=False, pad_y=False, shuffle=False)	
    train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

    if train==True:
        if gpu==False:
            tobiVO = Tobi_model()
        else:
            tobiVO = Tobi_model().cuda()
        # optimizer = optim.Adam(tobiVO.parameters(), lr=0.0001)
        optimizer = optim.Adam(tobiVO.parameters(), lr=0.00001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)
        run(model=tobiVO,data=train_dataset,batch=2,optimizer=optimizer,epoch=20,scheduler=scheduler)
    else:
        if gpu==False:
            tobiVO = Tobi_model()
        else:
            tobiVO = Tobi_model().cuda()
        tobiVO.load_state_dict(torch.load(model_path+'/modelsize0.pth')) # setting plz
        tobiVO.eval()

        #######################
        ### prediction part ###
        #######################
        a = np.array([0])
        for i in range(test_num):
            if gpu==False:
                input_ = torch.FloatTensor(train_dataset[i][1])
                input_ = torch.reshape(input_,(6,1,3,224,224))
                input_.requires_grad=False
            else:
                input_ = torch.FloatTensor(train_dataset[i][1]).cuda()
                input_ = torch.reshape(input_,(6,1,3,224,224)).cuda()
                input_.requires_grad=False
            for j in range(5):
                if(i==0 and j==0):
                    answer = tobiVO.forward(torch.cat([input_[j],input_[j+1]],dim = 0))
                    answer = answer.detach()
                    a = answer.numpy()
                else:
                    answer = tobiVO.forward(torch.cat([input_[j],input_[j+1]],dim = 0))
                    answer = answer.detach()
                    a = np.concatenate((a, answer.numpy()), axis=0)
            print('{} / {} '.format(i, test_num))
        np.save(predict_path+'x_predict.npy',a)
