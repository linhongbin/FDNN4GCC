import rospy
from sensor_msgs.msg import JointState

from std_msgs.msg import UInt8MultiArray

from std_msgs.msg import Bool
import dvrk
import numpy as np
from os.path import join
import torch
from Net import *
from loadModel import get_model, load_model
import time

from AnalyticalModel import *
import pdb
from Controller import Controller




# # #
# # # # # #
# # # # # # # #
# # # # # # #
# # # # # # # test controller function
MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
train_type = 'PKD'
model_type = 'DFNN'
#model_type = 'analytical_model'


controller = Controller(MTM_ARM)
controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)
# controller.load_gcc_model(model_type)
# pdb.set_trace()
time.sleep(1)
controller.move_MTM_joint(controller.GC_init_pos_arr)
time.sleep(4)
controller.start_gc()
#time.sleep(4)
# controller.stop_gc()
controller.ros_spin()
# controller.stop_gc()
# controller.move_MTM_joint(controller.GC_init_pos_arr)
# time.sleep(4)
# #
