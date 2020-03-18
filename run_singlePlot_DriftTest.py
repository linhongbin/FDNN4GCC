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
import os


fk_model = FK_MODEL()
################################################################################################################
save_pdf_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual",'result')
# define train and test path
drift_test_result_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual","archive","result2")
D = 6
sample_idx = 100
file_name_lst = ["analytical_model", "ReLU_Dual_UDirection_BP", "ReLU_Dual_UDirection_PKD"]

d_arr_lst = []
theta_arr_lst = []
for file_name in file_name_lst:
    drift_pos_tensor = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_tensor']
    drift_pos_cnt_arr = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_cnt_arr']
    sample_num = int(drift_pos_cnt_arr[0][sample_idx])
    drift_mat = drift_pos_tensor[drift_pos_tensor.shape[0]-sample_num:,:,sample_idx]
    drift_mat = np.flip(drift_mat, axis=0)
    d_arr = np.zeros(sample_num)
    theta_arr = np.zeros(sample_num)

    T_start = fk_model.forward(np.append(drift_mat[0,:], 0))
    for j in range(1,sample_num):
        T_end = fk_model.forward(np.append(drift_mat[j, :], 0))
        d = fk_model.transDiff(T_start, T_end)
        theta = fk_model.rotDiff(T_start, T_end)
        d_arr[j] = d*1000
        theta_arr[j] = np.degrees(theta)
        d_arr[j] = d
        theta_arr[j] = theta

    d_arr_lst.append(d_arr)
    theta_arr_lst.append(theta_arr)








legend_list = ['Model in [32]', 'DFNN with LfS', 'DFNN with PKD']


fig, ax = plt.subplots(figsize=(8, 4))
fontsize = 30
fill_color_list = ['tab:blue','tab:orange', 'tab:green']

for i in range(3):
    x = list(range(d_arr_lst[i].shape[0]))
    ax.plot(x, d_arr_lst[i].tolist(),color=fill_color_list[i], alpha=0.8, label=legend_list[i])

ax.yaxis.grid(True)

# maxValue = max([max(list) for list in abs_rms_list])

csfont = {'fontname':'Times New Roman'}

# Save the figure and show
ax.set_xlabel('time (s)',**csfont, fontsize=14)
ax.set_ylabel(r'Drift (Deg/mm)',**csfont, fontsize=14)

# plt.yscale('log',basey=10)

ax.margins(y=.1, x=.03)

plt.rcParams["font.family"] = "Times New Roman"

ax.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)

plt.tight_layout()
plt.show()
#
# if not os.path.exists(save_pdf_path):
#     os.makedirs(save_pdf_path)
#
#
# fig.savefig(join(save_pdf_path,'TrajTest_AbsRMS.pdf'),bbox_inches='tight')

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
