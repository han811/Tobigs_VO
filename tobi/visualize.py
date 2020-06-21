import matplotlib.pyplot as plt
import numpy as np
import time
import os
# from data_helper import get_data_info_tobi, SortedRandomBatchSampler, ImageSequenceDataset, ImageSequenceDataset2
# from params import par

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Tobi_model


# from tobi_util import ResNet, Bottleneck, ResNet_5 , ResNet_5_2
# from config import resnet_blk as r
# from model import Tobi_model

from torch.utils.tensorboard import SummaryWriter


# train_df = get_data_info_tobi(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=0, sample_times=par.sample_times)	
# # valid_df = get_data_info_tobi(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=5, sample_times=par.sample_times)
# train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
# train_dataset = ImageSequenceDataset2(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

# # translation y z x order
# def plot_route(gt, out, c_gt='g', c_out='r'):
# 	x_idx = 0
# 	y_idx = 2
# 	x = []
# 	y = []
# 	for i in range(len(gt)):
# 	# for i in range(100):
# 		if i%20==0:
# 			for j in range(6):
# 				x += [gt[i][2][j][x_idx].tolist()]
# 				y += [gt[i][2][j][y_idx].tolist()]
# 	plt.plot(x, y, color=c_gt, label='label')

# 	x = []
# 	y = []
# 	tempx = 0
# 	tempy = 0
# 	for i in range(len(predict)):
# 		tempx += out[i][x_idx]
# 		tempy += out[i][y_idx]
# 		x += [tempx]
# 		y += [tempy]
# 	plt.plot(x, y, color=c_out, label='predict')
# 	plt.gca().set_aspect('equal', adjustable='datalim')

# predict = []
# predict_path = os.getcwd()+'/result/x_predict.npy'
# predict = np.load(predict_path)
# plt.clf()
# plot_route(train_dataset, predict, 'r', 'b')
# plt.legend()
# plt.title('result')
# save_name = 'image/result_img2'
# plt.savefig(save_name)

model_path = '/home/kth/Desktop/deepVO/Tobigs_VO/tobi/modelsize19.pth'

x = np.load('/home/kth/Downloads/train.npy')
x_label = np.load('/home/kth/Downloads/train_label.npy')

tobiVO = Tobi_model()#.cuda()
tobiVO.load_state_dict(torch.load(model_path))
tobiVO.eval()
# answer = tobiVO.forward(torch.cat([x[0],x[1]],dim=0))
a = np.array([0])

# for i in range(x.shape[0]-1):
# print(x[1])
for i in range(20):
    x_1 = torch.FloatTensor(x[i])
    x_2 = torch.FloatTensor(x[i+1])

    # print(x_1)

    if i==0:
        x_1 = torch.unsqueeze(x_1,0)
        x_2 = torch.unsqueeze(x_2,0)

        answer = torch.FloatTensor(torch.cat([x_1,x_2],dim=0))
        answer = tobiVO.forward(answer)
        answer = answer.detach()
        a = answer.numpy()
    else:
        x_1 = torch.unsqueeze(x_1,0)
        x_2 = torch.unsqueeze(x_2,0)
        answer = torch.FloatTensor(torch.cat([x_1,x_2],dim=0))

        answer = tobiVO.forward(answer)
        answer = answer.detach()
        a = np.concatenate((a, answer.numpy()), axis=0)

x_idx=0
y_idx=2

x=[]
y=[]
d_1 = 0
d_2 = 0

for i in range(20):
    d_1+=x_label[i][x_idx]
    d_2+=x_label[i][y_idx]
    x+=[d_1]
    y+=[d_2]
plt.plot(x,y,color='g',label='label')

x=[]
y=[]
d_1 = 0
d_2 = 0
k_1=[]
k_2=[]


for i in range(20):
    # k_1 += [(x_label[i][x_idx]-a[i][x_idx])]
    # k_2 += [(x_label[i][y_idx]-a[i][y_idx])]
    d_1+=a[i][x_idx]
    d_2+=a[i][y_idx]
    x+=[d_1]
    y+=[d_2]
plt.plot(x,y,color='r',label='answer')
plt.xlim(-2,2)
plt.ylim(-2,2)
# print(sum(k_1))
plt.show()
# for i in range(len(x)):


# for i in range(test_num):
# 	if gpu==False:
# 		input_ = torch.FloatTensor(train_dataset[i][1])
# 		input_ = torch.reshape(input_,(6,1,3,224,224))
# 		input_.requires_grad=False
# 	else:
# 		input_ = torch.FloatTensor(train_dataset[i][1]).cuda()
# 		input_ = torch.reshape(input_,(6,1,3,224,224)).cuda()
# 		input_.requires_grad=False

# 	for j in range(5):
# 		if(i==0 and j==0):
# 			answer = tobiVO.forward(torch.cat([input_[j],input_[j+1]],dim = 0))
# 			answer = answer.detach()
# 			a = answer.numpy()
# 		else:
# 			answer = tobiVO.forward(torch.cat([input_[j],input_[j+1]],dim = 0))
# 			answer = answer.detach()
# 			a = np.concatenate((a, answer.numpy()), axis=0)
# np.save(predict_path,a)