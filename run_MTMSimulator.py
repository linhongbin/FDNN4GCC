import torch
from Net import *
import numpy as np
from sklearn import preprocessing
from AnalyticalModel import *
from os.path import join
from pathlib import Path
from scipy import io
from os import path



def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def create_simulator(save_dir, DistScale=1, sample_num=300, jntPosSensingNoise=1e-5, jntTorSensingNoise=1e-5):
    D = 6
    device = 'cpu'
    model_DistPos = SigmoidNet(D, 100, D).to(device)
    model_DistPos.apply(init_weights)
    model_DistNeg = SigmoidNet(D, 100, D).to(device)
    model_DistNeg.apply(init_weights)

    # sample the input space of the random function
    input_mat = np.zeros((sample_num, D))
    jnt_upper_limit = np.radians(np.array([40, 45, 34, 190, 175, 40]))
    jnt_lower_limit = np.radians(np.array([-40, -14, -34, -80, -85, -40]))
    for i in range(sample_num):
        rand_arr = np.random.rand(D)
        input_mat[i, :] = rand_arr * (jnt_upper_limit - jnt_lower_limit) + jnt_lower_limit

    # find the params for scaling the random functions to a unit output scale
    output_scaler_DistPos = preprocessing.StandardScaler().fit(model_DistPos(torch.from_numpy(input_mat).float()).detach().numpy())
    output_scaler_DistNeg = preprocessing.StandardScaler().fit(model_DistNeg(torch.from_numpy(input_mat).float()).detach().numpy())

    gravity_model = MTM_CAD()
    output_mat_gravity = gravity_model.predict(input_mat)


    avg_output_vec = np.abs(np.mean(output_mat_gravity, axis=0))
    avg_output_vec[0] = avg_output_vec[3]
    print(avg_output_vec)

    save_dict = {'model_DistPos': model_DistPos.state_dict()}
    save_dict['model_DistNeg'] = model_DistNeg.state_dict()
    save_dict['output_scaler_DistPos'] = output_scaler_DistPos
    save_dict['output_scaler_DistNeg'] = output_scaler_DistNeg
    save_dict['avg_output_vec'] = avg_output_vec
    save_dict['DistScale'] = DistScale
    save_dict['jntPosSensingNoise'] = jntPosSensingNoise
    save_dict['jntTorSensingNoise'] = jntTorSensingNoise


    save_dir = join(save_dir, 'Dist_'+str(DistScale))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, join(save_dir, 'simulator_param.pt'))




def generate_data(param_load_path, simulate_num, repetitive_num = 10, data_type='train'):
    data_type_list = ['train', 'test', 'validate']

    if data_type not in data_type_list:
        raise Exception("data_type should be", data_type_list)
    # load params of the simulator
    file = join(param_load_path, 'simulator_param.pt')
    if not path.isfile(file):
        raise Exception(file+ 'cannot not be found')
    checkpoint = torch.load(file)
    device = 'cpu'
    D = 6
    output_scaler_DistPos = checkpoint['output_scaler_DistPos']
    output_scaler_DistNeg = checkpoint['output_scaler_DistNeg']
    model_DistPos = SigmoidNet(D, 100, D).to(device)
    model_DistNeg = SigmoidNet(D, 100, D).to(device)
    model_DistPos.load_state_dict(checkpoint['model_DistPos'])
    model_DistNeg.load_state_dict(checkpoint['model_DistNeg'])
    avg_output_vec = checkpoint['avg_output_vec']
    DistScale = checkpoint['DistScale']
    jntPosSensingNoise = checkpoint['jntPosSensingNoise']
    jntTorSensingNoise = checkpoint['jntTorSensingNoise']

    # conduct the experiments repetitively
    for k in range(repetitive_num):

        # randomly sample the input space
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


        # random disturbance function with unit scale
        output_mat_DistPos = output_scaler_DistPos.transform(model_DistPos(torch.from_numpy(q_mat).float()).detach().numpy())
        output_mat_DistNeg = output_scaler_DistNeg.transform(model_DistNeg(torch.from_numpy(q_mat).float()).detach().numpy())


        gravity_model = MTM_CAD()

        # calculate the output matrix
        output_mat_gravity = gravity_model.predict(q_mat)
        for i in range(D):
            output_mat_DistPos[:,i] = output_mat_DistPos[:,i] * avg_output_vec[i]*DistScale
            output_mat_DistNeg[:, i] = output_mat_DistNeg[:, i] * avg_output_vec[i]*DistScale

        input_mat = np.concatenate((np.sin(q_mat), np.cos(q_mat), u_mat), axis=1)

        output_mat = output_mat_gravity + output_mat_DistPos*u_mat + output_mat_DistNeg *(1-u_mat)


        # add sensing noise to input for torques of training data
        if data_type == 'train':
            input_mat = input_mat + jntPosSensingNoise * (np.random.rand(input_mat.shape[0], input_mat.shape[1]) - 0.5) * 2
            output_mat = output_mat + jntTorSensingNoise * (np.random.rand(output_mat.shape[0],output_mat.shape[1])-0.5)*2


        # save simulated data
        save_dir = join(param_load_path, data_type, 'N'+str(simulate_num),'D6_SinCosInput')
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

        print("finish ",k+1," : (",k+1,"/",repetitive_num,")")





save_dir = join("data", "MTMR_28002", "sim", "random", 'N30000','D6_SinCosInput')

# DistScale = 0.02

DistScale = 1
simulate_num = 1000
repetitive_num = 4

# train_simulate_num_list = [100,500,1000,5000, 10000, 30000]
train_simulate_num_list = [10, 50, 100,500,1000, 5000]
test_simulate_num = 20000
save_dir = join("data", "MTMR_28002", "sim", "random", 'Dist_'+str(DistScale))

create_simulator(join("data", "MTMR_28002", "sim", "random"), DistScale=DistScale,sample_num=300)

for train_simulate_num in train_simulate_num_list:
    generate_data(save_dir, train_simulate_num, repetitive_num = repetitive_num, data_type='train')

generate_data(save_dir, test_simulate_num, repetitive_num=1, data_type='test')

generate_data(save_dir, 20000, repetitive_num=1, data_type='validate')