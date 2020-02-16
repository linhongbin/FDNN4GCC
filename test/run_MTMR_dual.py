import torch
from regularizeTool import EarlyStopping
from trainTool import multiTask_train
from Net import *
from loadDataTool import load_train_data
from os.path import join
from evaluateTool import testList
import scipy.io as sio
from os import mkdir
from loadModel import *

# path


def loop_func(use_net):
    # train_data_path = join("data", "Acrobot", "sim", train_file)
    # test_data_path = join("data", "Acrobot", "sim", "N34_std1")

    train_data_pathPos = join("data", "MTMR_28002",'real', 'uniform', 'D5N5','pos')
    train_data_pathNeg = join("data", "MTMR_28002",'real', 'uniform', 'D5N5','neg')
    train_data_pathList = [train_data_pathPos, train_data_pathNeg]
    test_data_path = join("data", "MTMR_28002", 'real', 'random', 'D5N10')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = 5
    if use_net == 'Lagrangian_SinNet':
        earlyStop_patience = 80
        learning_rate = 0.02
    else:
        earlyStop_patience = 40
        learning_rate = 0.02

    # config hyper-parameters
    max_training_epoch = 2000 # stop train when reach maximum training epoch
    goal_loss = 1e-4 # stop train when reach goal loss
    valid_ratio = 0.2 # ratio of validation data set over train and validate data
    batch_size = 256 # batch size for mini-batch gradient descent


    model = get_model('MTM', use_net, D, device=device)

    # load data
    print(device)
    print('FitNet:'+ use_net)
    train_loaderList = []
    valid_loaderList = []
    input_scalerList = []
    output_scalerList = []
    for i in range(len(train_data_pathList)):
        train_loader, valid_loader, input_scaler, output_scaler, input_dim, output_dim = load_train_data(data_dir=join(train_data_pathList[i], "data"),
                                                                                                         valid_ratio=valid_ratio,
                                                                                                         batch_size=batch_size,
                                                                                                         device=device)
        train_loaderList.append(train_loader)
        valid_loaderList.append(valid_loader)
        input_scalerList.append(input_scaler)
        output_scalerList.append(output_scaler)

    # configure network and optimizer
    # model = BPNet(input_dim, H, output_dim)

    loss_fn = torch.nn.SmoothL1Loss()
    modelParamList = list()
    for _model in model:
        modelParamList = modelParamList +list(_model.parameters())
    optimizer = torch.optim.Adam(modelParamList, lr=learning_rate, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)


    modelList = multiTask_train(model, train_loaderList, valid_loaderList, optimizer, loss_fn, early_stopping, max_training_epoch, is_plot=False)
    abs_rms_vec, rel_rms_vec = testList(modelList, test_data_path, input_scalerList, output_scalerList, device, verbose=True)

    save_model('.','test_dualController',modelList, input_scalerList, output_scalerList)
    # modelList, input_scalerList, output_scalerList = load_model('.','test',modelList)
    # print(input_scalerList)

#loop_func('N8_std1','Lagrangian_SinNet')
loop_func('Dual_Vanilla_SinSigmoidNet')


# test
#loop_func('N8_std1', 'Lagrangian_SinNet')