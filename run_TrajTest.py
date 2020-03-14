from Controller import Controller
import numpy as np
import time
import pdb
from os.path import join
import scipy.io
import os
import datetime

MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
load_testing_point_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual")
save_result_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual", "result")
train_type = 'BP'
# model_type = 'analytical_model'
model_type = 'DFNN'


D = 6
sample_num =10
controller = Controller(MTM_ARM)
q_mat, ready_q_mat = controller.random_testing_configuration(sample_num)

time.sleep(1)
sum_start_time = time.clock()

for i in range(sample_num):
    controller.move_MTM_joint(q_mat[i,:])
    time.sleep(2)

