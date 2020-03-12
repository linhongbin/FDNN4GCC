from Controller import Controller
import numpy as np
from os.path import join
import scipy.io
import os


MTM_ARM = 'MTMR'
save_testing_point_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual")

sample_num = 200
D = 6

controller = Controller(MTM_ARM)
q_mat, ready_q_mat = controller.random_testing_configuration(sample_num)




if not os.path.exists(save_testing_point_path):
    os.makedirs(save_testing_point_path)



file_name = "testing_points.mat"
scipy.io.savemat(join(save_testing_point_path, file_name), {'q_mat': q_mat,
                             'ready_q_mat': ready_q_mat})

