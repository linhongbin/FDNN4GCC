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


def create_simulator(save_dir,DistScale=1,sample_num=300):
    D = 6
    # DistScale = 0.8

    device = 'cpu'
    model_DistPos = SigmoidNet(D, 100, D).to(device)
    model_DistPos.apply(init_weights)
    model_DistNeg = SigmoidNet(D, 100, D).to(device)
    model_DistNeg.apply(init_weights)

    # sample_num = 300
    input_mat = np.zeros((sample_num, D))
    jnt_upper_limit = np.radians(np.array([40, 45, 34, 190, 175, 40]))
    jnt_lower_limit = np.radians(np.array([-40, -14, -34, -80, -85, -40]))
    for i in range(sample_num):
        rand_arr = np.random.rand(D)
        input_mat[i, :] = rand_arr * (jnt_upper_limit - jnt_lower_limit) + jnt_lower_limit

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
    Path(join(save_dir, 'Dist_'+str(DistScale))).mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, join(save_dir, 'Dist_'+str(DistScale),'simulator_param.pt'))


def generate_data(param_load_path, simulate_num):
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

    D = 6
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


    output_mat_DistPos = output_scaler_DistPos.transform(model_DistPos(torch.from_numpy(q_mat).float()).detach().numpy())
    output_mat_DistNeg = output_scaler_DistNeg.transform(model_DistNeg(torch.from_numpy(q_mat).float()).detach().numpy())


    gravity_model = MTM_CAD()
    output_mat_gravity = gravity_model.predict(q_mat)

    for i in range(D):
        output_mat_DistPos[:,i] = output_mat_DistPos[:,i] * avg_output_vec[i]*DistScale
        output_mat_DistNeg[:, i] = output_mat_DistNeg[:, i] * avg_output_vec[i]*DistScale

    input_mat = np.concatenate((np.sin(q_mat), np.cos(q_mat), u_mat), axis=1)

    output_mat = output_mat_gravity + output_mat_DistPos*u_mat + output_mat_DistNeg *(1-u_mat)

    print(output_mat)

    save_dir = join(param_load_path, 'random', 'N'+str(simulate_num),'D6_SinCosInput')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    io.savemat(join(save_dir, 'N'+str(simulate_num)+'_D6_SinCosInput.mat'),
               {'input_mat': input_mat, 'output_mat': output_mat})





save_dir = join("data", "MTMR_28002", "sim", "random", 'N30000','D6_SinCosInput')

create_simulator(join("data", "MTMR_28002", "sim", "random"),DistScale=0.1,sample_num=300)
generate_data(join("data", "MTMR_28002", "sim", "random", 'Dist_'+str(0.1)), 30000)