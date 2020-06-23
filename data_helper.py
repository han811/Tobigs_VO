#############################
### tobi - import package ###
#############################
import os
import glob
import time
import math
import pandas as pd
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R_

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from params import par
#################################
### tobi - import package end ###
#################################


################################################
### tobi - quaternion & translation def func ###
################################################
def R_to_quaternion(Rt):
    Rt = np.reshape(np.array(Rt), (3,4))
    t = Rt[:,-1]
    R = Rt[:,:3]
    r = R_.from_matrix(R)
    q = r.as_quat()
    pose_tovis = np.concatenate((t,q))
    return pose_tovis
####################################################
### tobi - quaternion & translation def func end ###
####################################################


###################################
### tobi - data_helper def func ###
###################################
def get_data_info_tobi(folder_list, overlap=False, pad_y=False, shuffle=False):
    X_path, Y = [], []
    for folder in folder_list:
        poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
        fpaths = sorted(glob.glob('{}{}/*.png'.format(par.image_dir, folder)))
        
        # image sequence length
        n_frames = len(fpaths)

        res = n_frames % 6
        if res != 0:
            n_frames = n_frames - res

        if overlap==False:
            x_segs = [fpaths[i:i+6] for i in range(0, n_frames, 6)]
            y_segs = [poses[i:i+6] for i in range(0, n_frames, 6)]
            Y += y_segs
            X_path += x_segs
        else:
            x_segs = [fpaths[i:i+6] for i in range(0, n_frames-5, 1)]
            y_segs = [poses[i:i+6] for i in range(0, n_frames-5, 1)]
            Y += y_segs
            X_path += x_segs

    # Convert to pandas dataframes
    data = {'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    return df

class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.image_arr = np.array(self.data_info.image_path)
        self.groundtruth_arr = np.array(self.data_info.pose)

    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        temp = np.ones((6,3))
        for i in range(6):
            for j in range(3):
                temp[i][j] = groundtruth_sequence[i][j]
        for i in range(6):
            for j in range(3):
                if i==0:
                    groundtruth_sequence[i][j] = groundtruth_sequence[i][j] - temp[i][j]
                else:
                    groundtruth_sequence[i][j] = groundtruth_sequence[i][j] - temp[i-1][j]
    
        image_path_sequence = self.image_arr[index]        
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
        return (image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.pose)

class ImageSequenceDataset_visual(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.image_arr = np.array(self.data_info.image_path)
        self.groundtruth_arr = np.array(self.data_info.pose)

    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        # temp = np.ones((6,3))
        # for i in range(6):
        #     for j in range(3):
        #         temp[i][j] = groundtruth_sequence[i][j]
        # for i in range(6):
        #     for j in range(3):
        #         if i==0:
        #             groundtruth_sequence[i][j] = groundtruth_sequence[i][j] - temp[i][j]
        #         else:
        #             groundtruth_sequence[i][j] = groundtruth_sequence[i][j] - temp[i-1][j]
    
        image_path_sequence = self.image_arr[index]        
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
        return (image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.pose)

#######################################
### tobi - data_helper def func end ###
#######################################

# Example of usage
if __name__ == '__main__':
    print("test")
