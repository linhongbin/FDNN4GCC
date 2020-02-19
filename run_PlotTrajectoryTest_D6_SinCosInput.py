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



################################################################################################################

# define train and test path
train_data_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual")
test_data_path = join("data", "MTMR_28002", "real", "random", 'N319','D6_SinCosInput')

# load Trajectory Test experiment data
test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu',input_scaler=None, output_scaler=None,
                             is_inputScale = False, is_outputScale = False)

test_input_mat = test_dataset.x_data.numpy()
test_ouput_mat = test_dataset.y_data.numpy()


test_output_hat_mat_List = []
legend_list = []
#
# # get predict CAD Model output
# MTM_CAD_model = MTM_CAD()
# test_output_hat_mat_List.append(MTM_CAD_model.predict(test_input_mat))
# legend_list.append('CAD')

# get predict MLSE4POL Model output
MTM_MLSE4POL_Model = MTM_MLSE4POL()
test_output_hat_mat_List.append(MTM_MLSE4POL_Model.predict(test_input_mat))
legend_list.append('MLSE4POL')


device = 'cpu'
D = 6
# get predict DNN with Knowledge Distillation output
use_net = 'ReLU_Dual_UDirection'
train_type = 'BP'
load_model_path = join(train_data_path, "result", "model")
model = get_model('MTM', use_net, D, device=device)
model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)
test_output_hat_mat_List.append(model.predict_NP(test_input_mat))
legend_list.append('DFNN with BP')


# get predict DNN with Knowledge Distillation output
use_net = 'ReLU_Dual_UDirection'
train_type = 'PKD'
load_model_path = join(train_data_path, "result", "model")
model = get_model('MTM', use_net, D, device=device)
model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)
test_output_hat_mat_List.append(model.predict_NP(test_input_mat))
legend_list.append('DFNN with PKD')



# plot predict error bar figures
abs_rms_list = []
rel_rms_list = []
mean_rel_rms_list = []
for i in range(len(test_output_hat_mat_List)):
    err_output_mat = np.abs(test_output_hat_mat_List[i] - test_ouput_mat)
    abs_rms_list.append(np.sqrt(np.mean(np.square(err_output_mat), axis=0)).tolist())
    rel_rms_list.append(np.sqrt(np.divide(np.mean(np.square(err_output_mat), axis=0),
                                    np.mean(np.square(test_ouput_mat), axis=0))).tolist())

for i in range(len(rel_rms_list)):
    abs_rms_list[i].append(np.mean(abs_rms_list[i], axis=0))
    rel_rms_list[i].append(np.mean(rel_rms_list[i],axis=0))


#print(err_output_mat)



jnt_index = np.arange(1,8)
fig, ax = plt.subplots()
w = 0.2
space = 0.2
capsize = 2
fontsize = 30

for i in range(len(abs_rms_list)):
    ax.bar(jnt_index+space*(i-1), abs_rms_list[i],  width=w,align='center', alpha=0.5, ecolor='black', capsize=capsize, label=legend_list[i])

ax.set_xticks(jnt_index)
labels = ['Joint '+str(i+1) for i in range(6)]
labels.append('Avg')
# ax.set_title('Absolute RMSE for Trajectory Test')
ax.yaxis.grid(True)
ax.autoscale(tight=True)
maxValue = max([max(list) for list in abs_rms_list])
plt.ylim(0, maxValue*1.2)

# Save the figure and show
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylabel(r'$\epsilon_{rms}$', fontsize=14)
ax.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
fig.savefig(join(train_data_path, "result",'TrajTest_AbsRMS.pdf'),bbox_inches='tight')



jnt_index = np.arange(1,8)
fig, ax = plt.subplots()
w = 0.2
space = 0.2
capsize = 2
fontsize = 30

for i in range(len(rel_rms_list)):
    ax.bar(jnt_index+space*(i-1), rel_rms_list[i],  width=w,align='center', alpha=0.5, ecolor='black', capsize=capsize, label=legend_list[i])

ax.set_xticks(jnt_index)
labels = ['Joint '+str(i+1) for i in range(6)]
labels.append('Avg')
# ax.set_title('Absolute RMSE for Trajectory Test')
ax.yaxis.grid(True)
ax.autoscale(tight=True)
maxValue = max([max(list) for list in rel_rms_list])
plt.ylim(0, maxValue*1.2)

# Save the figure and show
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylabel(r'$\epsilon_{rms}\%$', fontsize=14)
ax.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
fig.savefig(join(train_data_path, "result",'TrajTest_RelRMS.pdf'),bbox_inches='tight')
