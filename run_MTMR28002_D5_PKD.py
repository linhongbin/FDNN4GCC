from regularizeTool import EarlyStopping
from trainTool import KDtrain
from loadDataTool import load_train_N_validate_data, load_teacher_train_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, save_model
from AnalyticalModel import *
from HyperParam import get_hyper_param
def loop_func(train_data_path, valid_data_path, test_data_path, use_net,robot,teacherModelType):
    train_type = 'PKD'
    param_dict = get_hyper_param(robot,use_net=use_net,train_type=train_type)

    max_training_epoch = param_dict['max_training_epoch'] # stop train when reach maximum training epoch
    goal_loss = param_dict['goal_loss'] # stop train when reach goal loss
    batch_size = param_dict['batch_size'] # batch size for mini-batch gradient descent
    weight_decay = param_dict['weight_decay']
    device = param_dict['device']
    earlyStop_patience = param_dict['earlyStop_patience']
    learning_rate = param_dict['learning_rate']
    D = param_dict['D']
    teacher_sample_num = param_dict['teacher_sample_num']
    initLamda = param_dict['initLamda']
    endLamda = param_dict['endLamda']
    decayStepsLamda = param_dict['decayStepsLamda']

    device = torch.device(device)
    model = get_model('MTM', use_net, D, device=device)




    # train_loader, valid_loader, input_scaler, output_scaler, = load_train_N_validate_data(join(train_data_path, "data"),
    #                                                                                       batch_size, valid_data_path=None, valid_ratio=valid_ratio, device=device)
    train_loader, valid_loader, input_scaler, output_scaler, = load_train_N_validate_data(join(train_data_path, "data"),
                                                                                          batch_size,
                                                                                          valid_data_path=join(valid_data_path, "data"),
                                                                                          device=device,
                                                                                          input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = True)

    if teacherModelType == 'MTM_MLSE4POL':
        teacherModel = MTM_MLSE4POL()

    Teacher_trainLoader, _, _, _ = load_teacher_train_data(teacherModel, teacher_sample_num, batch_size, device,  input_scaler, output_scaler)

    # create train_loader from teacher model


    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)


    ### Train model
    model = KDtrain(model, train_loader, valid_loader, Teacher_trainLoader, optimizer, loss_fn, early_stopping, max_training_epoch, goal_loss, initLamda, endLamda, decayStepsLamda, is_plot=False)

    ### Get the predict output from test data and save to Matlab file
    train_dataset = load_data_dir(join(train_data_path,'data'), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    train_input_mat = train_dataset.x_data
    train_output_mat = train_dataset.y_data
    model = model.to('cpu')
    test_dataset = load_data_dir(join(test_data_path,"data"), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    test_input_mat = test_dataset.x_data
    test_output_mat = predict(model, test_input_mat, input_scaler, output_scaler)
    try:
        mkdir(join(train_data_path,"result"))
    except:
        print('Make directory: ', join(train_data_path,"result") + " already exist")

    # save data as .mat file
    save_result_path = join(train_data_path, "result", use_net+'.mat')
    print('Save result: ', save_result_path)
    sio.savemat(save_result_path, {'test_input_mat': test_input_mat.numpy(),
                                  'test_output_mat': test_output_mat.numpy(),
                                  'train_input_mat': train_input_mat.numpy(),
                                  'train_output_mat': train_output_mat.numpy()})

    test_loss, abs_rms_vec, rel_rms_vec = evaluate_rms(model, loss_fn, test_data_path, input_scaler, output_scaler, device, verbose=True)

    # save model to "result/model" folder
    model_save_path = join(train_data_path,"result","model")
    try:
        mkdir(model_save_path)
    except:
        print('Make directory: ', model_save_path + " already exist")
    save_model(model_save_path, use_net+'_KD_MLSE4POL', model, input_scaler, output_scaler)


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

# testMid_teacherModel
train_data_path = join("data", "MTMR_28002", "real", "uniform", "N5", 'D5', "dual")
valid_data_path = join("data", "MTMR_28002", "real", "uniform",  "N4", 'D5', "dual")
test_data_path = join("data", "MTMR_28002", "real", "random", 'N10','D5')
# loop_func(train_data_path, valid_data_path, test_data_path, 'SinNet')
loop_func(train_data_path, valid_data_path, test_data_path, 'SinInput_ReLUNet','MTMR28002',teacherModelType ='MTM_MLSE4POL')
