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


def run(model,data,label,batch,optimizer,epoch,scheduler):
    n = len(data)//batch
    r = len(data)%batch
    for j in range(epoch):
        for i in range(n):
            if i==(n-1):
                input = torch.tensor(data[i*batch:(i+1)*batch+r],requires_grad=True)
                input = torch.reshape(input,(6,1,3,224,224))
                # input = torch.tensor(data[i*batch:(i+1)*batch+r],requires_grad=True).cuda()
                # input = torch.reshape(input,(6,1,3,224,224)).cuda()
                label_ = label[i*batch:(i+1)*batch+r][0]
            else:
                input = torch.tensor(data[i*batch:(i+1)*batch],requires_grad=True)
                input = torch.reshape(input,(6,1,3,224,224))
                # input = torch.tensor(data[i*batch:(i+1)*batch],requires_grad=True).cuda()
                # input = torch.reshape(input,(6,1,3,224,224)).cuda()
                label_ = label[i*batch:(i+1)*batch][0]
            optimizer.zero_grad()
            loss = model.get_loss(input,torch.from_numpy(label_[1:6]))
            # loss = model.get_loss(input,torch.from_numpy(label_[1:6])).cuda())
            loss.backward(retain_graph = True)
            optimizer.step()
            print(loss)
        scheduler.step()
        torch.save(tobiVO.state_dict(),'./modelsize'+str(j)+'.pth')

data = np.load('/home/pjh/다운로드/data/data/x0.npy')
data_label = np.load('/home/pjh/다운로드/data/label/x_label0.npy')

# data_ = np.random.rand(1002,3,224,224)
# data_label_ = np.random.rand(1002,7)

data = np.reshape(data,(data.shape[0]//6,6,3,224,224))
data_label = np.reshape(data_label,(data_label.shape[0]//6,6,7))

print(data.dtype)
# tobiVO = Tobi_model()
tobiVO = Tobi_model()#.cuda()
#tobiVO.load_state_dict(torch.load('/home/pjh/Tobigs_VO/tobi/modelsize4.pth'))
#tobiVO.eval()
input = torch.tensor(data[100:101],requires_grad=True)#.cuda()
input = torch.reshape(input,(6,1,3,224,224))#.cuda()
label = torch.tensor(data_label[100:101][0][1])#.cuda()


answer = tobiVO.forward(torch.cat([input[0],input[1]],dim = 0))
optimizer = optim.Adam(tobiVO.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

if __name__ == "__main__":
    run(model=tobiVO,data=data,label=data_label,batch=1,optimizer=optimizer,epoch=20,scheduler=scheduler)
