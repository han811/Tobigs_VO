import matplotlib.pyplot as plt
import numpy as np
import time
import os
from data_helper import get_data_info_tobi, ImageSequenceDataset, ImageSequenceDataset_visual
from params import par

########################
### variable setting ###
########################

pose_GT_dir = par.pose_dir  #'KITTI/pose_GT/'
predicted_result_dir = os.getcwd()+'/image/'
predict_path = os.getcwd()+'/result/'

train_df = get_data_info_tobi(folder_list=par.train_video, overlap=False)	
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
train_dataset_gt = ImageSequenceDataset_visual(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
	
x_idx = 2
y_idx = 0

############################
### variable setting end ###
############################

def plot_route(gt, c_gt='g', c_out='r'):
	x = [v for v in gt[:, x_idx]]
	y = [v for v in gt[:, y_idx]]
	plt.plot(x, y, color=c_gt, label='Ground Truth')
	plt.savefig(predicted_result_dir+str(video))
	# plt.show()

def plot_route_predict(out, c_gt='g', c_out='r'):
	x_temp=0
	y_temp=0
	x = [v+x_temp for v in out[:, x_idx]]
	y = [v+y_temp for v in out[:, y_idx]]
	plt.plot(x, y, color=c_out, label='Predict')
	plt.savefig(predicted_result_dir+'predict')

if __name__=='__main__':
	# Load in GT and predicted pose
	video_list = ['00']
	for video in video_list:
		predict = np.load(predict_path+'x_predict.npy')
		GT_pose_path = '{}{}.npy'.format(pose_GT_dir, video)
		gt = np.load(GT_pose_path)
		plot_route(gt, c_gt='g', c_out='r')
		plt.close()
		plot_route_predict(predict, c_gt='g', c_out='r')
		plt.close()