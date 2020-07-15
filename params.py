import os

class Parameters():
	def __init__(self):
		
		############
		### Path ###
		############
		self.data_dir =  os.getcwd()+'/KITTI/'
		self.image_dir = self.data_dir + '/images/'
		self.pose_dir = self.data_dir + '/pose_GT/'
		

		#################
		### video num ###
		#################
		# self.train_video = ['00', '01', '02', '05', '08', '09']
		# self.train_video = ['01','04','07']
		self.train_video = ['07']
		# self.train_video = ['01','04']

		self.valid_video = ['04', '06', '07', '10']


		##########################
		### Data Preprocessing ###
		##########################
		self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
		self.img_w = 224   # original size is about 1226
		self.img_h = 224   # original size is about 370
		# self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
		self.img_means =  (-0.14965289656788702, -0.13016646662708822, -0.13578008882996825)
		# self.img_stds =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
		self.img_stds =  (0.31539906801272966, 0.319487843240816, 0.3234635824891331)
		self.minus_point_5 = True

par = Parameters()

