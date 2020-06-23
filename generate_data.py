import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from params import par
from data_helper import get_data_info_tobi, SortedRandomBatchSampler, ImageSequenceDataset

from tqdm import tqdm


train_df = get_data_info_tobi(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=0, sample_times=par.sample_times)	
valid_df = get_data_info_tobi(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=0, sample_times=par.sample_times)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
# valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
# valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

# print(len(train_dataset))

# for i in tqdm(range(0,len(train_dataset))):
for i in tqdm(range(0,400)):
    if i == 0:
        x = train_dataset[i][1].numpy()
        x_label = train_dataset[i][2].numpy()
    else:
        x = np.concatenate([x,train_dataset[i][1].numpy()],axis=0)
        x_label = np.concatenate([x_label,train_dataset[i][2].numpy()],axis=0)
np.save('train_data/train', x)
np.save('train_data/train_label', x_label)