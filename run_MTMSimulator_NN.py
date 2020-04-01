import torch
from Net import *
import numpy as np
from sklearn import preprocessing
from AnalyticalModel import *
from os.path import join
from pathlib import Path
from scipy import io
from os import path


# a template function to initialize the linear layer with randomized parameters
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# create parameters for a simulator
def create_simulator_param(save_dir, DistScale=1, sample_num=300, jntPosSensingNoise=1e-5, jntTorSensingNoise=1e-5):
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

    # get the params for scaling the random functions
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



# generate data for training, validating and testing
def generate_data(param_load_path, simulate_num, repetitive_num = 10, data_type='train'):
    data_type_list = ['train', 'validate', 'test']
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

        # generate data for gravity
        gravity_model = MTM_CAD()
        output_mat_gravity = gravity_model.predict(q_mat)

        # generate data for disturbance
        output_mat_DistPos = output_scaler_DistPos.transform(model_DistPos(torch.from_numpy(q_mat).float()).detach().numpy())
        output_mat_DistNeg = output_scaler_DistNeg.transform(model_DistNeg(torch.from_numpy(q_mat).float()).detach().numpy())
        for i in range(D):

            # disturbance = unit dist * average gravity output * Distscale
            output_mat_DistPos[:,i] = output_mat_DistPos[:,i] * avg_output_vec[i]*DistScale
            output_mat_DistNeg[:, i] = output_mat_DistNeg[:, i] * avg_output_vec[i]*DistScale

        # save as input and output matrix
        input_mat = np.concatenate((np.sin(q_mat), np.cos(q_mat), u_mat), axis=1)
        output_mat = output_mat_gravity + output_mat_DistPos*u_mat + output_mat_DistNeg *(1-u_mat)

        # add measuring white noise to input and output for torques for training data
        if data_type == 'train':
            input_mat = input_mat + jntPosSensingNoise * (np.random.rand(input_mat.shape[0], input_mat.shape[1]) - 0.5) * 2
            output_mat = output_mat + jntTorSensingNoise * (np.random.rand(output_mat.shape[0],output_mat.shape[1])-0.5)*2


        # save data
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
        print("finish "+data_type+" data ",k+1," : (",k+1,"/",repetitive_num,")")


# params for experiments
DistScale_lst = [0.02, 1]  # disturbance scale, w.r.t. output of gravity
train_repetitive_num = 10 # repetitive number for training data
train_simulate_num_list = [10, 50, 100,500,1000, 5000] # data amount for training data
validate_simulate_num = 20000 # data amount for validate data
test_simulate_num = 20000 # data amount for testing data


for DistScale in DistScale_lst:
    save_dir = join("data", "MTMR_28002", "sim", "random", 'NN_Dist_'+str(DistScale))

    # generate or update simulator param
    create_simulator_param(join("data", "MTMR_28002", "sim", "random"), DistScale=DistScale,sample_num=300)

    # generate training data
    for train_simulate_num in train_simulate_num_list:
        generate_data(save_dir, train_simulate_num, repetitive_num = train_repetitive_num, data_type='train')

    # generate validating data
    generate_data(save_dir, validate_simulate_num, repetitive_num=1, data_type='validate')

    # generate testing data
    generate_data(save_dir, test_simulate_num, repetitive_num=1, data_type='test')
