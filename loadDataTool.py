from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import os

class MTMDataset(Dataset):
    def __init__(self, data_list, device, input_scaler=None, output_scaler=None, is_inputScale = True, is_outputScale = True):
        input_mat = []
        output_mat = []
        # load .mat file to numpy
        for file_name in data_list:
            input = sio.loadmat(file_name)['input_mat']
            output = sio.loadmat(file_name)['output_mat']
            input_mat = input if len(input_mat)==0 else np.concatenate((input_mat, input), axis=0)
            output_mat = output if len(output_mat)==0 else np.concatenate((output_mat, output), axis=0)

        # scale output to zeroscore
        if is_inputScale:
            self.input_scaler = preprocessing.StandardScaler().fit(input_mat)
            input_mat = self.input_scaler.transform(input_mat)
        else:
            self.input_scaler = input_scaler
            if input_scaler is not None:
                input_mat = self.input_scaler.transform(input_mat)

        if is_outputScale:
            self.output_scaler = preprocessing.StandardScaler().fit(output_mat)
            output_mat = self.output_scaler.transform(output_mat)
        else:
            self.output_scaler = output_scaler
            if output_scaler is not None:
                output_mat = self.output_scaler.transform(output_mat)

        # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device).float()
        self.y_data = torch.from_numpy(output_mat).to(device).float()

        # get length of pair CAD_sim_1e6
        self.len = self.x_data.shape[0]

        # get dimension of input and output CAD_sim_1e6
        self.input_dim = input_mat.shape[1]
        self.output_dim = output_mat.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class NumpyDataSet(Dataset):
    def __init__(self, input_mat, output_mat, device):
        # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device).float()
        self.y_data = torch.from_numpy(output_mat).to(device).float()

        # get length of pair CAD_sim_1e6
        self.len = self.x_data.shape[0]

        # get dimension of input and output CAD_sim_1e6
        self.input_dim = input_mat.shape[1]
        self.output_dim = output_mat.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def load_data_dir(data_dir, device, input_scaler=None, output_scaler=None, is_inputScale = True, is_outputScale = True):
    data_list = []
    if not os.path.exists(data_dir):
        raise Exception('cannot find directory: {}'.format(os.getcwd()+data_dir))
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                print('Load Data: ', os.path.join(root, file))
                data_list.append(os.path.join(root, file))
    full_dataset = MTMDataset(data_list, device, input_scaler, output_scaler, is_inputScale, is_outputScale)
    return full_dataset

def load_train_N_validate_data(train_data_dir, batch_size, valid_data_path=None, valid_ratio=0.2,device='cpu',
                               input_scaler=None, output_scaler=None, is_inputScale = True, is_outputScale = True):
    if valid_data_path == None:
        full_dataset = load_data_dir(train_data_dir, device, input_scaler, output_scaler, is_inputScale, is_outputScale)
        train_ratio = 1 - valid_ratio
        train_size = int(full_dataset.__len__() * train_ratio)
        test_size = full_dataset.__len__() - train_size
        train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        valid_loader = DataLoader(validate_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)
        input_scaler = full_dataset.input_scaler
        output_scaler = full_dataset.output_scaler

    else:
        train_dataset = load_data_dir(train_data_dir, device, input_scaler, output_scaler, is_inputScale, is_outputScale)
        valid_dataset = load_data_dir(valid_data_path, device, input_scaler, output_scaler, is_inputScale, is_outputScale)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)
        input_scaler = train_dataset.input_scaler
        output_scaler = train_dataset.output_scaler

    return train_loader, valid_loader, input_scaler, output_scaler

def load_train_N_validate_data_list(train_data_dir_list, batch_size, valid_data_path_list=None, valid_ratio=0.2,device='cpu'):
    train_loaderList = []
    valid_loaderList = []
    input_scalerList = []
    output_scalerList = []
    if valid_data_path_list == None:
        for i in range(len(train_data_dir_list)):
            full_dataset = load_data_dir(train_data_dir_list[i], device)
            train_ratio = 1 - valid_ratio
            train_size = int(full_dataset.__len__() * train_ratio)
            test_size = full_dataset.__len__() - train_size
            train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      shuffle=True
                                      )
            valid_loader = DataLoader(validate_dataset,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      shuffle=True)
            train_loaderList.append(train_loader)
            valid_loaderList.append(valid_loader)
            input_scalerList.append(full_dataset.input_scaler)
            output_scalerList.append(full_dataset.output_scaler)
    else:
        for i in range(len(train_data_dir_list)):
            train_dataset = load_data_dir(train_data_dir_list[i], device)

            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      shuffle=True
                                      )
            train_loaderList.append(train_loader)

            input_scalerList.append(train_dataset.input_scaler)
            output_scalerList.append(train_dataset.output_scaler)

        for i in range(len(valid_data_path_list)):
            valid_dataset = load_data_dir(valid_data_path_list[i], device)
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      shuffle=True)
            valid_loaderList.append(valid_loader)
    return train_loaderList, valid_loaderList, input_scalerList, output_scalerList

def load_train_data(data_dir, valid_ratio, batch_size, device):
    full_dataset = load_data_dir(data_dir, device)
    train_ratio = 1 - valid_ratio
    train_size = int(full_dataset.__len__() * train_ratio)
    test_size = full_dataset.__len__() - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True
                              )
    valid_loader = DataLoader(validate_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True)

    return train_loader, valid_loader, full_dataset.input_scaler, full_dataset.output_scaler, full_dataset.input_dim, full_dataset.output_dim


def load_teacher_train_data(teacherModel, sample_num, batch_size, device, input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False, train_ratio=1):
    input_mat, output_mat, input_scaler, output_scaler = teacherModel.random_model_sampling(sample_num, input_scaler, output_scaler,  is_inputScale, is_outputScale)
    dataset = NumpyDataSet(input_mat, output_mat, device)
    if train_ratio==1:
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        valid_loader = None
    else:
        train_size = int(dataset.__len__() * train_ratio)
        test_size = dataset.__len__() - train_size
        train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        valid_loader = DataLoader(validate_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)


def load_preProcessData(train_data_dir, batch_size, device, valid_data_path = None, valid_ratio = 0.2,teacherModel=None, teacher_sample_num=None):

    # load sampled train data
    D = 6
    data_list = []
    if not os.path.exists(train_data_dir):
        raise Exception('cannot find directory: {}'.format(os.getcwd()+train_data_dir))
    for root, dirs, files in os.walk(train_data_dir):
        for file in files:
            if file.endswith(".mat"):
                print('Load Data: ', os.path.join(root, file))
                data_list.append(os.path.join(root, file))
    train_input_mat = []
    train_output_mat = []
    # load .mat file to numpy
    for file_name in data_list:
        input = sio.loadmat(file_name)['input_mat']
        output = sio.loadmat(file_name)['output_mat']
        train_input_mat = input if len(train_input_mat) == 0 else np.concatenate((train_input_mat, input), axis=0)
        train_output_mat = output if len(train_output_mat) == 0 else np.concatenate((train_output_mat, output), axis=0)

    # load data of teacher model
    if teacherModel is not None:
        input_mat, output_mat, _, _ = teacherModel.random_model_sampling(teacher_sample_num, None,None,False,False)
        teacher_input_mat = []
        teacher_output_mat = []
        # load .mat file to numpy
        for file_name in data_list:
            input = sio.loadmat(file_name)['input_mat']
            output = sio.loadmat(file_name)['output_mat']
            teacher_input_mat = input if len(teacher_input_mat) == 0 else np.concatenate((teacher_input_mat, input), axis=0)
            teacher_output_mat = output if len(teacher_output_mat) == 0 else np.concatenate((teacher_output_mat, output), axis=0)


    # caculate mean and std
    total_input_mat = train_input_mat
    total_output_mat = train_output_mat
    if teacherModel is not None:
        total_input_mat = np.concatenate((total_input_mat, teacher_input_mat), axis=0)
        total_output_mat = np.concatenate((total_output_mat, teacher_output_mat), axis=0)

    input_mean = np.mean(total_input_mat[:,:D*2], axis=0)
    output_mean = np.mean(total_output_mat, axis=0)
    input_std = np.std(total_input_mat[:,:D*2], axis=0)
    output_std = np.std(total_output_mat, axis=0)

    # scaling the input and output matrix
    for i in range(D*2):
        train_input_mat[i,:D*2] = (train_input_mat[i,:D*2] - input_mean[i])/input_std[i]
    for i in range(D):
        train_output_mat[i,:] = (train_output_mat[i,:] - output_mean[i])/output_std[i]
    train_dataset = NumpyDataSet(train_input_mat, train_output_mat, device)


    # scaling the input and output matrix
    if teacherModel is not None:
        for i in range(D*2):
            teacher_input_mat[i, :D * 2] = (teacher_input_mat[i, :D * 2] - input_mean[i]) / input_std[i]
        for i in range(D):
            teacher_output_mat[i, :] = (teacher_output_mat[i, :] - output_mean[i]) / output_std[i]
        teacher_dataset = NumpyDataSet(teacher_input_mat, teacher_output_mat, device)

    if valid_data_path == None:
        train_ratio = 1 - valid_ratio
        train_size = int(train_dataset.__len__() * train_ratio)
        test_size = train_dataset.__len__() - train_size
        train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        valid_loader = DataLoader(validate_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )

        data_list = []
        if not os.path.exists(valid_data_path):
            raise Exception('cannot find directory: {}'.format(os.getcwd() + valid_data_path))
        for root, dirs, files in os.walk(valid_data_path):
            for file in files:
                if file.endswith(".mat"):
                    print('Load Data: ', os.path.join(root, file))
                    data_list.append(os.path.join(root, file))
        valid_input_mat = []
        valid_output_mat = []
        # load .mat file to numpy
        for file_name in data_list:
            input = sio.loadmat(file_name)['input_mat']
            output = sio.loadmat(file_name)['output_mat']
            valid_input_mat = input if len(valid_input_mat) == 0 else np.concatenate((valid_input_mat, input), axis=0)
            valid_output_mat = output if len(valid_output_mat) == 0 else np.concatenate((valid_output_mat, output),
                                                                                        axis=0)


        # scaling the input and output matrix
        for i in range(D*2):
            valid_input_mat[i, :D * 2] = (valid_input_mat[i, :D * 2] - input_mean[i]) / input_std[i]
        for i in range(D):
            valid_output_mat[i, :] = (valid_output_mat[i, :] - output_mean[i]) / output_std[i]

        valid_dataset = NumpyDataSet(valid_input_mat, valid_output_mat, device)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)
    if teacherModel is not None:
        teacher_loader = DataLoader(teacher_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True)
    else:
        teacher_loader = None




    return train_loader, valid_loader, teacher_loader, input_mean, input_std, output_mean, output_std
