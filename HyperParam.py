import torch
def get_hyper_param(robot, use_net=None, train_type=None):
    if robot == 'MTMR28002':
        param_dict = {}
        param_dict['max_training_epoch'] = 2000 # stop train when reach maximum training epoch
        param_dict['goal_loss'] = 1e-4 # stop train when reach goal loss
        param_dict['valid_ratio'] = 0.2 # ratio of validation data set over train and validate data
        param_dict['batch_size'] = 256 # batch size for mini-batch gradient descent
        param_dict['weight_decay'] = 1e-4
        param_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        param_dict['earlyStop_patience'] = 20
        param_dict['learning_rate'] = 0.06
        param_dict['D'] = 5

        if train_type == 'PKD':
            param_dict['teacher_sample_num'] = 30000
            param_dict['initLamda'] = 5
            param_dict['endLamda'] = 0.01
            param_dict['decayStepsLamda'] = 100


    return param_dict