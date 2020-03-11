from Controller import Controller
import numpy as np
import time
import pdb
from os.path import join

MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
train_type = 'BP'
model_type = 'DFNN'



controller = Controller(MTM_ARM)
controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)
q_mat, ready_q_mat = controller.random_testing_configuration(100)

q_mat = np.concatenate((q_mat, np.zeros((q_mat.shape[0], 1))), axis=1)
ready_q_mat = np.concatenate((ready_q_mat, np.zeros((ready_q_mat.shape[0], 1))), axis=1)

# for j in range(6):
#     print("---------- ")
#     print("jnt_upper_limit: ", np.degrees(controller.jnt_upper_limit[j]))
#     print("jnt_lower_limit: ", np.degrees(controller.jnt_lower_limit[j]))
#     for i in range(7):
#         print("q_mat :", np.degrees(q_mat[i, j]))
#         print("ready_q_mat :",np.degrees(ready_q_mat[i, j]))
#
# print("++++++++++++")
# print("jnt_coup_upper_limit: ", np.degrees(controller.jnt_coup_upper_limit))
# print("jnt_coup_lower_limit: ", np.degrees(controller.jnt_coup_lower_limit))
# for i in range(7):
#     print("q_mat :", np.degrees(q_mat[i, 1]+q_mat[i, 2]))
#     print("ready_q_mat :", np.degrees(ready_q_mat[i, 1]+ready_q_mat[i, 2]))
#pdb.set_trace()

# pdb.set_trace()
rate = 100
duration = 3
controller.FIFO_buffer_size = rate * duration
for i in range(4):
    time.sleep(0.3)
    controller.move_MTM_joint(controller.GC_init_pos_arr)
    time.sleep(0.1)
    controller.move_MTM_joint(ready_q_mat[i,:])
    time.sleep(0.1)
    controller.move_MTM_joint(q_mat[i,:])
    time.sleep(0.5)

    controller.clear_FIFO_buffer()
    controller.start_gc()

    isExceedSafeVel = False
    while not (controller.FIFO_pos_cnt==rate*duration or isExceedSafeVel):
        isExceedSafeVel = controller.isExceedSafeVel

    print("isExceedSafeVel: ", isExceedSafeVel)
    print("Buffer: ", controller.FIFO_pos)
    print("Buffer count ", controller.FIFO_pos_cnt)
    controller.stop_gc()
    controller.move_MTM_joint(controller.GC_init_pos_arr)
    time.sleep(0.5)

