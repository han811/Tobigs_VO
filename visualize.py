import matplotlib.pyplot as plt
import numpy as np
import time
import os
from data_helper import get_data_info_tobi, SortedRandomBatchSampler, ImageSequenceDataset, ImageSequenceDataset2
from params import par


train_df = get_data_info_tobi(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=5, sample_times=par.sample_times)	
# valid_df = get_data_info_tobi(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=5, sample_times=par.sample_times)
train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset2(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

# translation y z x order
def plot_route(gt, out, c_gt='g', c_out='r'):
	x_idx = 2
	y_idx = 0
	x = []
	y = []
	for i in range(len(gt)):
	# for i in range(100):
		if i%20==0:
			for j in range(6):
				x += [gt[i][2][j][x_idx].tolist()]
				y += [gt[i][2][j][y_idx].tolist()]
		print(i)
	plt.plot(x, y, color=c_gt, label='label')

	x = []
	y = []
	tempx = 0
	tempy = 0
	for i in range(len(predict)):
		tempx += out[i][x_idx]
		tempy += out[i][y_idx]
		x += [tempx]
		y += [tempy]
	plt.plot(x, y, color=c_out, label='predict')
	plt.gca().set_aspect('equal', adjustable='datalim')

predict = []
# predict_path = os.getcwd()+'/result/x_predict.npy'
# predict = np.load(predict_path)
plt.clf()
plot_route(train_dataset, predict, 'r', 'b')
plt.legend()
plt.title('result')
save_name = 'image/result_img'
plt.savefig(save_name)

