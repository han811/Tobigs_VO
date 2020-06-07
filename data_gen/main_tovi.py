import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info, get_data_info_tovi

from tqdm import tqdm

# train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)	
# valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
train_df = get_data_info_tovi(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=0, sample_times=par.sample_times)	
valid_df = get_data_info_tovi(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=0, sample_times=par.sample_times)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
# train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)
# print(train_dataset[3007][2].numpy())

# valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
# valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
# # print(len(valid_dataset))
# # valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

# torch.reshape(train_dataset,train_dataset.size)
# train = list()
# for i in range(train_dataset.groundtruth_arr.shape[0]):



n = train_dataset.groundtruth_arr.shape[0]//3
r = train_dataset.groundtruth_arr.shape[0]%3
for j in range(3):
	if j==2:
		for i in tqdm(range(n*j,n*(j+1)+r)):
			if i == n*j:
				x = train_dataset[i][1].numpy()
				x_label = train_dataset[i][2].numpy()
			else:
				x = np.concatenate([x,train_dataset[i][1].numpy()],axis=0)
				x_label = np.concatenate([x_label,train_dataset[i][2].numpy()],axis=0)

	else:
		for i in tqdm(range(n*j,n*(j+1))):
			if i == n*j:
				x = train_dataset[i][1].numpy()
				x_label = train_dataset[i][2].numpy()
			else:
				x = np.concatenate([x,train_dataset[i][1].numpy()],axis=0)
				x_label = np.concatenate([x_label,train_dataset[i][2].numpy()],axis=0)
	np.save('x'+str(j), x)
	np.save('x_label'+str(j), x_label)
# for i in tqdm(range(valid_dataset.groundtruth_arr.shape[0])):
# 	if i == 0:
# 		y = valid_dataset[i][1]
# 		y_label = valid_dataset[i][2]
# 	else:
# 		torch.cat([y,valid_dataset[i][1]],dim=0)
# 		torch.cat([y_label,valid_dataset[i][2]],dim=0)
# np.save('y', y)
# np.save('y_label', y_label)