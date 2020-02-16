import torch
import numpy as np
import torch.utils.data as data
from sklearn import preprocessing
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import os


class MTMDataset(Dataset):
    # Initialize your CAD_sim_1e6, download, etc.
    def __init__(self, data_list, device='cpu'):
        input_mat = []
        output_mat = []
        # load .mat CAD_sim_1e6 to numpy
        for file_name in data_list:
            input = sio.loadmat(file_name)['input_mat']
            output = sio.loadmat(file_name)['output_mat']
            input_mat = input if len(input_mat)==0 else np.concatenate((input_mat, input), axis=0)
            output_mat = output if len(output_mat)==0 else np.concatenate((output_mat, output), axis=0)

        # _, s_size = input_mat.shape
        # indices = list(range(s_size))
        # np.random.shuffle(indices)
        # input_mat = input_mat[:, indices[:train_samples_num]]
        # output_mat = output_mat[:, indices[:train_samples_num]]

        # CAD_sim_1e6 pre-processing
            # scale output to zeroscore
        self.output_scaler = preprocessing.StandardScaler().fit(output_mat)
        output_mat = self.output_scaler.transform(output_mat)
        # input feature = [sin(q) cos(q)]
        input_mat = np.concatenate((np.sin(input_mat), np.cos(input_mat)), axis=1)
            # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device)
        self.y_data = torch.from_numpy(output_mat).to(device)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


data_list = [];
for root, dirs, files in os.walk(os.path.join("CAD_sim_1e6","CAD_sim_1e6")):
    for file in files:
        if file.endswith(".mat"):
            print(os.path.join(root, file))
            data_list.append(os.path.join(root, file))

full_dataset = MTMDataset(data_list)
train_ratio = 0.8
train_size = int(full_dataset.__len__()*train_ratio)
test_size = full_dataset.__len__() - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
batch_size = 256
train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=True
                        )
valid_loader = DataLoader(validate_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=True)


# def load_train_data(train_input_file_list, train_output_file_list,valid_size=0.2,batch_size=128, device='cpu'):
#     input_mat = []
#     output_mat = []
#     # load .mat CAD_sim_1e6 to numpy
#     for train_input_file, train_output_file in zip(train_input_file_list, train_output_file_list):
#         input = sio.loadmat(pjoin('CAD_sim_1e6',train_input_file))['input_mat']
#         output = sio.loadmat(pjoin('CAD_sim_1e6',train_output_file))['output_mat']
#         input_mat = input if len(input_mat)==0 else np.concatenate((input_mat, input), axis=1)
#         output_mat = output if len(output_mat)==0 else np.concatenate((output_mat, output), axis=1)
#
#     # _, s_size = input_mat.shape
#     # indices = list(range(s_size))
#     # np.random.shuffle(indices)
#     # input_mat = input_mat[:, indices[:train_samples_num]]
#     # output_mat = output_mat[:, indices[:train_samples_num]]
#
#     # CAD_sim_1e6 pre-processing
#     input_mat = input_mat.T
#     output_mat = output_mat.T
#     input_mat = input_mat[:, :-1]
#     output_mat = output_mat[:, :-1]
#         # scale output to zeroscore
#     output_scaler = preprocessing.StandardScaler().fit(output_mat)
#     output_mat = output_scaler.transform(output_mat)
#     # input feature = [sin(q) cos(q)]
#     input_mat = np.concatenate((np.sin(input_mat), np.cos(input_mat)), axis=1)
#         # numpy to torch tensor
#     x = torch.from_numpy(input_mat).to(device)
#     y = torch.from_numpy(output_mat).to(device)
#     x = x.float()
#     y = y.float()
#     (N_in, D_in) = x.shape
#     (N_out, D_out) = y.shape
#     assert N_in == N_out
#
#     # split train set and vaidate set
#     indices = list(range(N_in))
#     np.random.shuffle(indices)
#     split = int(np.floor(valid_size * N_in))
#     train_idx, valid_idx = indices[split:], indices[:split]
#     train_sampler = CAD_sim_1e6.sampler.SubsetRandomSampler(train_idx)
#     valid_sampler = CAD_sim_1e6.sampler.SubsetRandomSampler(valid_idx)
#     train_dataSet = CAD_sim_1e6.TensorDataset(x, y)
#     train_loader = torch.utils.CAD_sim_1e6.DataLoader(train_dataSet,
#                                                batch_size=batch_size,
#                                                sampler=train_sampler,
#                                                num_workers=0)
#     valid_loader = torch.utils.CAD_sim_1e6.DataLoader(train_dataSet,
#                                                batch_size=batch_size,
#                                                sampler=valid_sampler,
#                                                num_workers=0)
#
#     return train_loader, valid_loader, output_scaler