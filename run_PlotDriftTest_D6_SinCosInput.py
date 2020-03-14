from regularizeTool import EarlyStopping
from trainTool import train
from loadDataTool import load_train_N_validate_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from AnalyticalModel import *
import numpy as np


fk_model = FK_MODEL()
################################################################################################################

# define train and test path
drift_test_result_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual","archive","result1")
D = 6


drift_mat_lst = []

cartes_dirft_mat_lst = []
mean_arr_lst = []
file_name_lst = ["analytical_model", "ReLU_Dual_UDirection_BP", "ReLU_Dual_UDirection_PKD"]
for file_name in file_name_lst:
    drift_pos_tensor = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_tensor']
    drift_pos_cnt_arr = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_cnt_arr']
    sample_num = drift_pos_tensor.shape[2]
    drift_mat = np.zeros((sample_num, D+2 ))
    for i in range(sample_num):
        start_jnt_arr = drift_pos_tensor[-1,:,i]
        end_jnt_arr = drift_pos_tensor[0,:,i]
        drift_mat[i,:D] = end_jnt_arr - start_jnt_arr

        T_start = fk_model.forward(np.append(start_jnt_arr, 0))
        T_end = fk_model.forward(np.append(end_jnt_arr, 0))
        d = fk_model.transDiff(T_start, T_end)
        theta = fk_model.rotDiff(T_start, T_end)
        drift_mat[i,D] = d
        drift_mat[i, D+1] = theta

    # print(drift_mat)
    drift_mat_lst.append(drift_mat)
    mean_arr_lst.append(np.mean(drift_mat, axis=0))
    print(mean_arr_lst[-1])


# for i in range(len(mean_arr_lst)):
#     mean_arr_lst[i] = np.append(mean_arr_lst[i], np.mean(mean_arr_lst[i]))
#
#print(mean_arr_lst)






legend_list = ['Physical Teacher Model', 'DFNN with LfS', 'DFNN with PKD']

#
# # plot predict error bar figures
# abs_rms_list = []
# rel_rms_list = []
# mean_rel_rms_list = []
# for i in range(len(test_output_hat_mat_List)):
#     err_output_mat = np.abs(test_output_hat_mat_List[i] - test_ouput_mat)
#     abs_rms_list.append(np.sqrt(np.mean(np.square(err_output_mat), axis=0)).tolist())
#     rel_rms_list.append(np.sqrt(np.divide(np.mean(np.square(err_output_mat), axis=0),
#                                     np.mean(np.square(test_ouput_mat), axis=0))).tolist())
#
# for i in range(len(rel_rms_list)):
#     abs_rms_list[i].append(np.mean(abs_rms_list[i], axis=0))
#     rel_rms_list[i].append(np.mean(rel_rms_list[i],axis=0))
#
# for i in range(len(rel_rms_list)):
#     rel_rms_list[i] =[k*100 for k in rel_rms_list[i]]
#
# #print(err_output_mat)
#
#
#
# jnt_index = np.arange(1,8)
# fig, ax = plt.subplots()
# w = 0.2
# space = 0.2
# capsize = 2
# fontsize = 30
# fill_color_list = ['tab:blue','tab:orange', 'tab:green']
#
# for i in range(len(abs_rms_list)):
#     ax.bar(jnt_index+space*(i-1), abs_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.6, ecolor='black', capsize=capsize, label=legend_list[i])
#
# ax.set_xticks(jnt_index)
# labels = ['Joint '+str(i+1) for i in range(6)]
# labels.append('Avg')
# # ax.set_title('Absolute RMSE for Trajectory Test')
# ax.yaxis.grid(True)
# ax.autoscale(tight=True)
# maxValue = max([max(list) for list in abs_rms_list])
# plt.ylim(0, maxValue*1.2)
#
# # Save the figure and show
# ax.set_xticklabels(labels, fontsize=14)
# ax.set_ylabel(r'$\epsilon_{rms}$', fontsize=14)
# ax.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.tight_layout()
# plt.show()
# fig.savefig(join(train_data_path, "result",'TrajTest_AbsRMS.pdf'),bbox_inches='tight')
#
#
#
# jnt_index = np.arange(1,8)
# fig, ax = plt.subplots()
# w = 0.2
# space = 0.2
# capsize = 2
# fontsize = 30
#
# for i in range(len(rel_rms_list)):
#     ax.bar(jnt_index+space*(i-1), rel_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.6, ecolor='black', capsize=capsize, label=legend_list[i])
#
# ax.set_xticks(jnt_index)
# labels = ['Joint '+str(i+1) for i in range(6)]
# labels.append('Avg')
# # ax.set_title('Absolute RMSE for Trajectory Test')
# ax.yaxis.grid(True)
# ax.autoscale(tight=True)
# maxValue = max([max(list) for list in rel_rms_list])
# plt.ylim(0, maxValue*1.2)
#
# # Save the figure and show
# ax.set_xticklabels(labels, fontsize=14)
# ax.set_ylabel(r'$\epsilon_{rms}\%$', fontsize=14)
# ax.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.tight_layout()
# plt.show()
# fig.savefig(join(train_data_path, "result",'TrajTest_RelRMS.pdf'),bbox_inches='tight')
#
#
# print('Avg Absolute RMSE: ',[lst[-1] for lst in abs_rms_list])
# print('Avg Relative RMSE: ',[lst[-1] for lst in rel_rms_list])
