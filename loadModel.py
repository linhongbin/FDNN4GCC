from Net import *
import numpy as np
from os import path, mkdir
import torch
def get_model(robot, use_net, D, device='cpu'):
    # define net for MTM
    if robot == 'MTM':
        if use_net == 'SinNet':
            model = SinNet(D, 400, D).to(device)
        elif use_net == 'VanillaSin_SigmoidNet':
            base_model = SinNet(D, 400, D).to(device)
            additon_model = SigmoidNet(D, 30, D).to(device)
            model = VanillaNet(base_model, additon_model)
        elif use_net == 'VanillaSin_ReluNet':
            base_model = SinNet(D, 400, D).to(device)
            additon_model = ReLuNet(D, [100], D).to(device)
            model = VanillaNet(base_model, additon_model)
        elif use_net == 'ReLuNet':
            model = ReLuNet(D, [30,30,30], D).to(device)
        elif use_net == 'SigmoidNet':
            model = SigmoidNet(D, 100, D).to(device)
        elif use_net == 'Multi_SinNet':
            model = Multi_SinNet(D, 100, D).to(device)
        elif use_net == 'Dual_Vanilla_SinSigmoidNet':
            base_model = SinNet(D, 500, D).to(device)
            additon_modelPos = SigmoidNet(D, 20, D).to(device)
            additon_modelNeg = SigmoidNet(D, 20, D).to(device)
            modelPos = VanillaNet(base_model, additon_modelPos)
            modelNeg = VanillaNet(base_model, additon_modelNeg)
            model = [modelPos, modelNeg]
        elif use_net == 'Dual_SinNet':
            modelPos = SinNet(D, 500, D).to(device)
            modelNeg = SinNet(D, 500, D).to(device)
            model = [modelPos, modelNeg]
        elif use_net == 'VanillaBPNet':
            base_model = SinNet(D, 100, D).to(device)
            additon_modelPos = SigmoidNet(D, 20, D).to(device)
            additon_modelNeg = SigmoidNet(D, 20, D).to(device)
            modelPos = VanillaNet(base_model, additon_modelPos)
            modelNeg = VanillaNet(base_model, additon_modelNeg)
            model = [modelPos, modelNeg]
        elif use_net == 'Lagrangian_SinNet':
            base_model = SinNet(D, 300, 1).to(device)
            # delta_q = 1e-2
            # w_list = [1, 1, 3, 3, 3]
            # w_vec = torch.from_numpy(np.array(w_list).T).to(device).float()
            model = LagrangeNet(base_model)
        elif use_net == 'KDNet_Parallel':
            DOF, K_VecNum, Com_LayerList, K_LayerList, D_LayerList = 5, 21, [20,20], [20],[20,10]
            model = KDNet_Parallel(DOF, K_VecNum, Com_LayerList, K_LayerList, D_LayerList).to(device)
        elif use_net == 'KDNet_Serial':
            DOF, K_LayerList, D_LayerList, K_VecNum = 5, [20], [20, 20], 21
            model = KDNet_Serial(DOF, K_LayerList, D_LayerList, K_VecNum).to(device)
        elif use_net == 'Two_ReLuNet':
            K_Vec_DIM = 21
            model1 = ReLuNet(D, [30,30,30], K_Vec_DIM).to(device)
            model2 = ReLuNet(K_Vec_DIM, [30, 30, 30], D).to(device)
            model = torch.nn.Sequential(model1, model2)
        elif use_net == 'SinInput_ReLUNet':
            model = SinInput_ReLUNet(D, [30,30,30], D)

        elif use_net == 'ReLU_Dual_UDirection':
            pos_net = ReLuNet(D, [30, 30, 30], D).to(device)
            neg_net = ReLuNet(D, [30, 30, 30], D).to(device)
            model = DualNets_UDirection(pos_net, neg_net, D, device)
        else:
            raise Exception(use_net + 'is not support')
    ### define net for acrobot
    elif robot == 'Acrobot':
        if use_net == 'SinNet':
            model = SinNet(D, 100, D).to(device)
        elif use_net == 'ReLuNet':
            model = ReLuNet(D, [100], D).to(device)
        elif use_net == 'SigmoidNet':
            model = SigmoidNet(D, 100, D).to(device)
        elif use_net == 'Multi_SinNet':
            model = Multi_SinNet(D, 100, D).to(device)
        elif use_net == 'VanillaNet':
            base_model = SinNet(D, 100, D).to(device)
            additon_model = PolNet(2, 4).to(device)
            model = VanillaNet(base_model, additon_model)
        elif use_net == 'Lagrangian_SinNet':
            base_model = SinNet(D, 300, 1).to(device)
            # delta_q = 1e-2
            # w_list = [1, 1]
            # w_vec = torch.from_numpy(np.array(w_list).T).to(device).float()
            model = LagrangeNet(base_model)
        elif use_net == 'VanillaSinPol_Net':
            base_model = SinNet(D, 100, D).to(device)
            additon_model = PolNet(2, 4).to(device)
            model = VanillaNet(base_model, additon_model)
        elif use_net == 'VanillaSinSigmoid_Net':
            base_model = SinNet(D, 100, D).to(device)
            additon_model = SigmoidNet(D, 10, D).to(device)
            model = VanillaNet(base_model, additon_model)
        elif use_net == 'SinLogNet':
            model = SinLogNet(D, 100, D).to(device)
        else:
            raise Exception(use_net + 'is not support')
    else:
        raise Exception(robot + 'is not support')


    return model

def save_model(file_path, file_name, model, input_scaler=None, output_scaler=None):
    if not path.exists(file_path):
        mkdir(path)

    if isinstance(model, list):
        save_dict = {'model' + str(i + 1): model[i].state_dict() for i in range(len(model))}
    else:
        save_dict = {'model': model.state_dict()}

    if input_scaler is not None:
        save_dict['input_scaler'] = input_scaler
    if output_scaler is not None:
        save_dict['output_scaler'] = output_scaler

    torch.save(save_dict, path.join(file_path, file_name+'.pt'))

def load_model(file_path, file_name, model):
    file = path.join(file_path, file_name)
    file = file+'.pt'
    if not path.isfile(file):
        raise Exception(file+ 'cannot not be found')

    checkpoint = torch.load(file)
    if isinstance(model, list):
        for i in range(len(model)):
            model[i].load_state_dict(checkpoint['model' + str(i + 1)])
    else:
        model.load_state_dict(checkpoint['model'])

    if 'input_scaler' in checkpoint:
        input_scaler = checkpoint['input_scaler']
    else:
        input_scaler = None

    if 'output_scaler' in checkpoint:
        output_scaler = checkpoint['output_scaler']
    else:
        output_scaler = None
    return model, input_scaler, output_scaler