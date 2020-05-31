import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info


# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
	f.write('\n'+'='*50 + '\n')
	f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
	f.write('\n'+'='*50 + '\n')

# Prepare Data
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
	print('Load data info from {}'.format(par.train_data_info_path))
	train_df = pd.read_pickle(par.train_data_info_path)
	valid_df = pd.read_pickle(par.valid_data_info_path)
else:
	print('Create new data info')
	if par.partition != None:
		partition = par.partition
		train_df, valid_df = get_partition_data_info(partition, par.train_video, par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=True)
	else:
		train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)	
		valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
	# save the data info
	train_df.to_pickle(par.train_data_info_path)
	valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('='*50)









