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
from model import Tobi_model

import os
from params import par
from data_helper import get_data_info_tobi, SortedRandomBatchSampler, ImageSequenceDataset

train=True
gpu=False
predict_path = os.getcwd()+'/result/x_predict.npy'
model_path = os.getcwd()+'/weight'
# data_path = '/home/kth/Desktop/tovis-git/Tobigs_VO/tobi/data/data/x0.npy'
# data_label_path = '/home/kth/Desktop/tovis-git/Tobigs_VO/tobi/data/label/x_label0.npy'
test_num=10

# should fix for when batch != 1
def run(model,data,optimizer,epoch,scheduler,batch=1):
    n = len(data)//batch
    r = len(data)%batch
    for j in range(epoch):
        for i in range(n):
            if gpu==False:
                input = torch.FloatTensor(data[i][1])
                input.requires_grad=True
                input = torch.reshape(input,(6,1,3,224,224))
                label_ = torch.FloatTensor(data[i][2])
            else:
                input = torch.FloatTensor(data[i][1])
                input.requires_grad=True
                input = torch.reshape(input,(6,1,3,224,224)).cuda()
                label_ = torch.FloatTensor(data[i][2]).cuda()
            optimizer.zero_grad()
            loss = model.get_loss(input,label_[1:6])
            loss.backward(retain_graph=True)
            optimizer.step()
            print(loss)
        torch.save(tobiVO.state_dict(),model_path+'/modelsize'+str(j)+'.pth')

if __name__ == "__main__":
    train_df = get_data_info_tobi(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=5, sample_times=par.sample_times)	
    # valid_df = get_data_info_tobi(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=5, sample_times=par.sample_times)
    train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
    train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

    if train==True:
        if gpu==False:
            tobiVO = Tobi_model()
        else:
            tobiVO = Tobi_model().cuda()
        optimizer = optim.Adam(tobiVO.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)
        run(model=tobiVO,data=train_dataset,batch=1,optimizer=optimizer,epoch=20,scheduler=scheduler)
    else:
        if gpu==False:
            tobiVO = Tobi_model()
        else:
            tobiVO = Tobi_model().cuda()
        tobiVO.load_state_dict(torch.load(model_path))
        tobiVO.eval()

        a = np.array([0])
        for i in range(test_num):
            if gpu==False:
                input = torch.FloatTensor(train_dataset[i][1])
                input = torch.reshape(input,(6,1,3,224,224))
                input.requires_grad=False
            else:
                input = torch.FloatTensor(train_dataset[i][1]).cuda()
                input = torch.reshape(input,(6,1,3,224,224)).cuda()
                input.requires_grad=False
            for j in range(5):
                if(i==0 and j==0):
                    answer = tobiVO.forward(torch.cat([input[j],input[j+1]],dim = 0))
                    answer = answer.detach()
                    a = answer.numpy()
                else:
                    answer = tobiVO.forward(torch.cat([input[j],input[j+1]],dim = 0))
                    answer = answer.detach()
                    a = np.concatenate((a, answer.numpy()), axis=0)
        np.save(predict_path,a)