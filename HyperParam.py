import torch
def get_hyper_param(robot, use_net=None, train_type=None, is_sim = False):
    if robot == 'MTMR28002':
        param_dict = {}
        param_dict['max_training_epoch'] = 2000 # stop train when reach maximum training epoch
        param_dict['goal_loss'] = 1e-4 # stop train when reach goal loss
        param_dict['valid_ratio'] = 0.2 # ratio of validation data set over train and validate data
        param_dict['batch_size'] = 256 # batch size for mini-batch gradient descent
        param_dict['weight_decay'] = 1e-4
        param_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        param_dict['earlyStop_patience'] = 30
        param_dict['learning_rate'] = 0.06
        param_dict['D'] = 6

        if train_type == 'PKD':
            if not is_sim:
                param_dict['learning_rate'] = 0.01
                param_dict['teacher_sample_num'] = 30000
                param_dict['initLamda'] = 2
                param_dict['endLamda'] = 1.5
                param_dict['decayStepsLamda'] = 30
            else:
                param_dict['learning_rate'] = 0.01
                param_dict['teacher_sample_num'] = 300
                param_dict['initLamda'] = 0.002
                param_dict['endLamda'] = 0.0001
                param_dict['decayStepsLamda'] = 30


    return param_dict