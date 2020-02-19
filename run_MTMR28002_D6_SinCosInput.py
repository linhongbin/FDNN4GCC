from regularizeTool import EarlyStopping
from trainTool import train,KDtrain
from loadDataTool import load_preProcessData
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, save_model
from HyperParam import get_hyper_param
from AnalyticalModel import *

def loop_func(train_data_path, valid_data_path, test_data_path, use_net, robot, train_type='BP'):
    param_dict = get_hyper_param(robot, train_type=train_type)

    max_training_epoch = param_dict['max_training_epoch'] # stop train when reach maximum training epoch
    goal_loss = param_dict['goal_loss'] # stop train when reach goal loss
    batch_size = param_dict['batch_size'] # batch size for mini-batch gradient descent
    weight_decay = param_dict['weight_decay']
    device = param_dict['device']
    earlyStop_patience = param_dict['earlyStop_patience']
    learning_rate = param_dict['learning_rate']
    D = param_dict['D']

    device = torch.device(device)
    model = get_model('MTM', use_net, D, device=device)
    if train_type == 'BP':
        train_loader, valid_loader, _, input_mean, input_std, output_mean, output_std =load_preProcessData(join(train_data_path, "data"),
                                                                                                           batch_size,
                                                                                                           device,
                                                                                                           valid_data_path=join(valid_data_path, "data"))
    elif train_type == 'PKD':
        teacherModel = MTM_MLSE4POL()
        train_loader, valid_loader, teacher_loader, input_mean, input_std, output_mean, output_std = load_preProcessData(join(train_data_path, "data"),
                                                                                                                        batch_size,
                                                                                                                        device,
                                                                                                                        valid_data_path=join(valid_data_path, "data"),
                                                                                                                        teacherModel=teacherModel,
                                                                                                                        teacher_sample_num=param_dict['teacher_sample_num'])


    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

    ### Train model
    model.set_normalized_param(input_mean, input_std, output_mean, output_std)
    if train_type=='BP':
        model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch, goal_loss, is_plot=False)
    elif train_type == 'PKD':
        model = KDtrain(model, train_loader, valid_loader, teacher_loader, optimizer, loss_fn, early_stopping,
                        max_training_epoch, goal_loss, param_dict['initLamda'], param_dict['endLamda'], param_dict['decayStepsLamda'], is_plot=False)
    else:
        raise Exception("cannot recoginze the train type")
    #
    # ### Get the predict output from test data and save to Matlab file
    # train_dataset = load_data_dir(join(train_data_path,'data'), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    # train_input_mat = train_dataset.x_data
    # train_output_mat = train_dataset.y_data
    # model = model.to('cpu')
    # test_dataset = load_data_dir(join(test_data_path,"data"), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    # test_input_mat = test_dataset.x_data
    # test_output_mat = predict(model, test_input_mat, input_scaler, output_scaler)
    # try:
    #     mkdir(join(train_data_path,"result"))
    # except:
    #     print('Make directory: ', join(train_data_path,"result") + " already exist")
    #
    # # save data as .mat file
    # save_result_path = join(train_data_path, "result", use_net+'.mat')
    # print('Save result: ', save_result_path)
    # sio.savemat(save_result_path, {'test_input_mat': test_input_mat.numpy(),
    #                               'test_output_mat': test_output_mat.numpy(),
    #                               'train_input_mat': train_input_mat.numpy(),
    #                               'train_output_mat': train_output_mat.numpy()})
    #
    # test_loss, abs_rms_vec, rel_rms_vec = evaluate_rms(model, loss_fn, test_data_path, input_scaler, output_scaler, device, verbose=True)

    # save model to "result/model" folder
    test_dataset = load_data_dir(join(test_data_path, "data"), device=device, input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    feature_mat = test_dataset.x_data.numpy()
    target_mat = test_dataset.y_data.numpy()
    target_hat_mat = model.predict_NP(feature_mat)

    rel_rms_vec = np.sqrt(np.divide(np.mean(np.square(target_hat_mat - target_mat), axis=0),
                                    np.mean(np.square(target_mat), axis=0)))

    abs_rms_vec = np.sqrt(np.mean(np.square(target_hat_mat - target_mat), axis=0))

    print('Absolute RMS for each joint are:', abs_rms_vec)
    print('Relative RMS for each joint are:', rel_rms_vec)


    model_save_path = join(train_data_path,"result","model")
    try:
        mkdir(model_save_path)
    except:
        print('Make directory: ', model_save_path + " already exist")
    save_model(model_save_path, use_net+'_'+train_type, model)


################################################################################################################


# N_list = [2,3,4,5,6,7,8,9,10,12,15,17,20]
# std = 1
# train_file_list = ['N'+str(i)+'_std'+str(std) for i in N_list]
#
# use_net_list = ['SinNet', 'ReLuNet', 'SigmoidNet','Lagrangian_SinNet']


# for use_net in use_net_list:
#     train_data_path = join("data", "MTMR_28002", "real", "uniform","D5N5","dual")
#     test_data_path = join("data", "MTMR_28002", "real", "random","D5N10")
#     loop_func(train_data_path, test_data_path, use_net)

# test
train_data_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual")
valid_data_path = join("data", "MTMR_28002", "real", "uniform",  "N5", 'D6_SinCosInput', "dual")
test_data_path = join("data", "MTMR_28002", "real", "random", 'N319','D6_SinCosInput')

loop_func(train_data_path, valid_data_path, test_data_path, 'ReLU_Dual_UDirection','MTMR28002', train_type='BP')
loop_func(train_data_path, valid_data_path, test_data_path, 'ReLU_Dual_UDirection','MTMR28002', train_type='PKD')


