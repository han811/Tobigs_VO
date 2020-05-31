#!/usr/bin/env python

import rospy
import rospkg
from sensor_msgs.msg import Image
from matplotlib import pyplot as plt
import numpy as np

####################
### opencv usage ###
####################
# from cv_bridge import CvBridge
# import cv2

class Img_saver:
    def __init__(self,im_topic):
        self.im_topic = im_topic
        self.im_sub = rospy.Subscriber(self.im_topic,Image,self.im_callback)
        self.data = Image()
        self.im_buffer = Image()
        self.im = Image()
        self.im_width = 640
        self.im_height = 480

        self.trigger = False

        self.np_im = 0

        self.rate

        ####################
        ### opencv usage ###
        ####################
        # self.br = CvBridge()
        # self.im_cv = 0

    def im_callback(self,data):
        self.data = data
        self.im_buffer = [data.header.stamp,data.data]
        self.im_width = data.width
        self.im_height = data.height

        self.trigger = True

    def get_im(self):
        if self.trigger == True:
            self.im = self.im_buffer
    
    ####################
    ### opencv usage ###
    ####################
    # def get_cv_im(self):
    #     self.im_cv = self.br.imgmsg_to_cv2(self.data)

    def get_np_im(self):
        if self.trigger == True:
            self.np_im = np.fromstring(self.im[1],dtype=np.uint8)

    def save_im(self):
        self.get_im()
        self.get_np_im()
        # plt.imsave(path,self.np_im)

    def spin(self,rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()



if __name__=='__main__':
    rospy.init_node('save_data', anonymous=True)
    topic_name = rospy.get_param('im_topic')
    im_saver = Img_saver(topic_name)
    try:
        print(len(im_saver.np_im))
        im_saver.spin(10)
    except rospy.ROSInterruptException:
        pass