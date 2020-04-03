import torch
from Net import *
import numpy as np
from sklearn import preprocessing
from AnalyticalModel import *
from os.path import join
from pathlib import Path
from scipy import io
from os import path



# generate data for training, validating and testing
def generate_data(save_path, simulate_num, repetitive_num = 10, data_type='train', jntPosSensingNoise=1e-5, jntTorSensingNoise=1e-5):
    data_type_list = ['train', 'validate', 'test']
    if data_type not in data_type_list:
        raise Exception("data_type should be", data_type_list)


    D = 6

    # repetitive experiments
    for k in range(repetitive_num):

        # randomly sample q and u(delta q) randomly
        q_mat = np.zeros((simulate_num, D))
        u_mat = np.zeros((simulate_num, D))
        jnt_upper_limit = np.radians(np.array([40, 45, 34, 190, 175, 40]))
        jnt_lower_limit = np.radians(np.array([-40, -14, -34, -80, -85, -40]))
        for i in range(simulate_num):
            rand_arr = np.random.rand(D)
            q_mat[i, :] = rand_arr * (jnt_upper_limit - jnt_lower_limit) + jnt_lower_limit
        for i in range(simulate_num):
            rand_arr = np.random.rand(D)
            u = np.zeros(D)
            u[rand_arr>0.5] = 1
            u_mat[i,:] = u

        # generate data for gravity with disturbance
        model = MTM_MLSE4POL()
        input_mat = np.concatenate((np.sin(q_mat), np.cos(q_mat), u_mat), axis=1)
        output_mat = model.predict(input_mat)

        # add measuring white noise to input and output for torques for training data
        if data_type == 'train' or data_type == 'validate':
            input_mat = input_mat + jntPosSensingNoise * (np.random.rand(input_mat.shape[0], input_mat.shape[1]) - 0.5) * 2
            output_mat = output_mat + jntTorSensingNoise * (np.random.rand(output_mat.shape[0],output_mat.shape[1])-0.5)*2


        # save data
        save_dir = join(save_path, data_type, 'N'+str(simulate_num),'D6_SinCosInput')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        sample_index = 1
        while path.isdir(join(save_dir, str(sample_index))):
            sample_index += 1
        if data_type == 'train':
            save_dir = join(save_dir, str(sample_index))
        save_dir = join(save_dir, 'data')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        io.savemat(join(save_dir, 'N'+str(simulate_num)+'_D6_SinCosInput.mat'),
                   {'input_mat': input_mat, 'output_mat': output_mat})
        print("finish "+data_type+" data ",k+1," : (",k+1,"/",repetitive_num,")")

# create teacher model parameters
def create_TM_param(root_path, param_noise_scale):
    model = MTM_MLSE4POL()
    param_vec = model.param_vec
    param_vec[10:,:] = param_vec[10:,:] + 2 * (np.random.rand(60,1)-0.5) * param_noise_scale
    save_dict = {'TM_param_noise_scale': param_noise_scale}
    save_dict['TM_param_vec'] = param_vec
    Path(root_path).mkdir(parents=True, exist_ok=True)
    io.savemat(join(root_path, 'simulation_param.mat'), save_dict)


# params for experiments
root_dir = join("data", "MTMR_28002", "sim", "random", 'MLSE4POL')
train_repetitive_num = 10 # repetitive number for training data
train_simulate_num_list = [10, 50, 100,500,1000, 5000] # data amount for training data
validate_simulate_num = 20000 # data amount for validate data
test_simulate_num = 20000 # data amount for testing data
jntPosSensingNoise=1e-5 # noise for measuring positional signals
jntTorSensingNoise=1e-5 # noise for measuring torque signals
experiment_sets_num = 2
param_noise_scale_lst = [1e-3, 4e-3]

for k in range(experiment_sets_num):
    save_dir = join(root_dir, str(k+1))

    # save simulation param
    create_TM_param(save_dir, param_noise_scale=param_noise_scale_lst[k])

    # generate training data
    for train_simulate_num in train_simulate_num_list:
        generate_data(save_dir, train_simulate_num, repetitive_num = train_repetitive_num, data_type='train', jntPosSensingNoise=jntPosSensingNoise, jntTorSensingNoise = jntTorSensingNoise)

    # generate validating data
    generate_data(save_dir, validate_simulate_num, repetitive_num=1, data_type='validate', jntPosSensingNoise=jntPosSensingNoise, jntTorSensingNoise = jntTorSensingNoise)

    # generate testing data
    generate_data(save_dir, test_simulate_num, repetitive_num=1, data_type='test', jntPosSensingNoise=jntPosSensingNoise, jntTorSensingNoise = jntTorSensingNoise)


