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

def run(model,data,label,batch,optimizer,epoch):
    n = len(data)//batch
    r = len(data)%batch
    for j in range(epoch):
        for i in range(n):
            if i==(n-1):
                input = torch.tensor(data[i*batch:(i+1)*batch+r],requires_grad=True)
                input = torch.reshape(input,(6,1,3,224,224))
                label_ = label[i*batch:(i+1)*batch+r][0]
            else:
                input = torch.tensor(data[i*batch:(i+1)*batch],requires_grad=True)
                input = torch.reshape(input,(6,1,3,224,224))
                label_ = label[i*batch:(i+1)*batch][0]
            optimizer.zero_grad()
            loss = model.get_loss(input,torch.from_numpy(label_[1:6]))
            loss.backward(retain_graph = True)
            optimizer.step()
            print(loss)
            if i%100:
                torch.save(tobiVO,'./modelsize'+str(i))
data = np.load('/home/kth/Desktop/tovis-git/Tobigs_VO/tobi/data/data/x0.npy')
data_label = np.load('/home/kth/Desktop/tovis-git/Tobigs_VO/tobi/data/label/x_label0.npy')
data = np.reshape(data,(data.shape[0]//6,6,3,224,224))
data_label = np.reshape(data_label,(data_label.shape[0]//6,6,7))

data = data[0:1]
data_label = data_label[0:1]



tobiVO = Tobi_model()
optimizer = optim.Adam(tobiVO.parameters(), lr=0.0001)

if __name__ == "__main__":
    run(model=tobiVO,data=data,label=data_label,batch=1,optimizer=optimizer,epoch=20)
