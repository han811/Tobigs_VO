######################################################
### tobi - quaternion & translation import package ###
######################################################
import numpy as np
import math
from scipy.spatial.transform import Rotation as R_
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0
##########################################################
### tobi - quaternion & translation import package end ###
##########################################################


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

################################
### translation + quaternion ###
################################
	
def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle
####################################################
### tobi - quaternion & translation def func end ###
####################################################