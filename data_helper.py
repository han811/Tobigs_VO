#############################
### tobi - import package ###
#############################
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
from params import par
from helper import normalize_angle_delta
#################################
### tobi - import package end ###
#################################

###################################
### tobi - data_helper def func ###
###################################
def get_data_info_tobi(folder_list, seq_len_range=6, overlap=5, sample_times=1, pad_y=False, shuffle=False):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
        fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, folder))
        # Fixed seq_len

        seq_len = seq_len_range
        # image sequence length
        n_frames = len(fpaths)
        jump = seq_len - overlap
        # jump to the next image
        res = n_frames % seq_len
        if res != 0:
            n_frames = n_frames - res
        x_segs = [fpaths[i:i+seq_len] for i in range(0, n_frames-overlap, jump)]
        y_segs = [poses[i:i+seq_len] for i in range(0, n_frames-overlap, jump)]
        Y += y_segs
        X_path += x_segs
        X_len += [len(xs) for xs in x_segs]
        # each image group length
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    return df

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=True):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = self.df.iloc[:].seq_len.unique()
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_sample_list = [i for i in range(n_sample)]
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            tmp = [n_sample_list[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.array(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.array(self.data_info.pose)

    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        temp = np.array(groundtruth_sequence[0])
        for i in range(6):
            for j in range(3):
                groundtruth_sequence[i][j] = groundtruth_sequence[i][j] - temp[j]
        # print(groundtruth_sequence)
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

class ImageSequenceDataset2(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.array(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.array(self.data_info.pose)

    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        temp = np.array(groundtruth_sequence[0])
        # print(groundtruth_sequence)
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)



# Example of usage
if __name__ == '__main__':
    print("test")

