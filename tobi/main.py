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

data = np.load('/home/kth/Desktop/tovis-git/Tobigs_VO/tobi/data/x.npy')
data_label = np.load('/home/kth/Desktop/tovis-git/Tobigs_VO/tobi/data/x_label.npy')
data_label = torch.tensor(data_label[1:6])
new_input = torch.tensor(data[0:6])
new_input = torch.reshape(new_input,(6,1,3,224,224))
print(new_input.size())

tobiVO = Tobi_model()
optimizer = optim.Adam(tobiVO.parameters(), lr=0.0005)
# torch.save(tobiVO,'./modelsize')
optimizer.zero_grad()
# tobiVO(new_input)
loss = tobiVO.get_loss(new_input,data_label)
loss.backward(retain_graph = True)
optimizer.step()
print(loss)