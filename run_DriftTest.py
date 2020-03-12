from Controller import Controller
import numpy as np
import time
import pdb
from os.path import join
import scipy.io
import os


MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
load_testing_point_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual")
save_result_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual", "result")
train_type = 'PKD'
model_type = 'DFNN'


D = 6

controller = Controller(MTM_ARM)
controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)

q_mat = scipy.io.loadmat(join(load_testing_point_path, 'testing_points.mat'))['q_mat']
ready_q_mat = scipy.io.loadmat(join(load_testing_point_path, 'testing_points.mat'))['ready_q_mat']

sample_num = q_mat.shape[0]

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


if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)


# pdb.set_trace()
rate = 500
duration = 2.5
controller.FIFO_buffer_size = rate * duration

drift_pos_tensor = np.zeros((rate * duration, D, sample_num))
drift_time = np.zeros((sample_num))
drift_pos_cnt_arr = np.zeros((sample_num))
drift_isExceedSafeVel_arr = np.full((sample_num), True, dtype=bool)



for i in range(sample_num):
    controller.move_MTM_joint(ready_q_mat[i,:])
    controller.move_MTM_joint(q_mat[i,:])
    time.sleep(0.5)

    controller.clear_FIFO_buffer()
    controller.start_gc()

    isExceedSafeVel = False
    # while True:
    start = time.clock()
    controller.isExceedSafeVel =False
    while not (controller.FIFO_pos_cnt==rate*duration or isExceedSafeVel):
        # print(controller.FIFO_pos_cnt)
        time.sleep(0.001)
        isExceedSafeVel = controller.isExceedSafeVel

    drift_pos_tensor[:,:,i] = controller.FIFO_pos
    drift_isExceedSafeVel_arr[i] = isExceedSafeVel
    drift_pos_cnt_arr[i] = controller.FIFO_pos_cnt

    elapsed = time.clock()
    elapsed = elapsed - start
    drift_time[i] = elapsed
    print "Time spent in (function name) is: ", elapsed


    print("isExceedSafeVel: ", isExceedSafeVel)
    # print("Buffer: ", controller.FIFO_pos)
    # print("Buffer count ", controller.FIFO_pos_cnt)

    print("finish ("+str(i+1)+"/"+str(sample_num)+")")
    controller.stop_gc()
    controller.move_MTM_joint(controller.GC_init_pos_arr)

file_name = use_net+'_'+train_type
scipy.io.savemat(join(save_result_path, file_name), {'drift_pos_tensor': drift_pos_tensor,
                             'drift_isExceedSafeVel_arr': drift_isExceedSafeVel_arr,
                             'drift_pos_cnt_arr': drift_pos_cnt_arr,
                             'drift_time':drift_time})

