from os.path import join as pjoin
import scipy.io as sio
import torch
import _pickle as cPickle
from regularizeTool import EarlyStopping
from trainTool import load_train_data, train, test
from Net import LogNet


# global configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
H = 1000 # number of hidden neurons
learning_rate = 0.01 # learning rate
max_training_epoch = 2000 # maximum training epoch
goal_loss = 1e-3 # goal loss
valid_size = 0.2
batch_size = 256
earlyStop_patience = 8
model_file_name = 'LogNet_CAD_sim_rand_1e5_Source'
train_input_file_list = ['CAD_sim_rand_1e5_pos.mat']
train_output_file_list = ['CAD_sim_rand_1e5_tor.mat']
test_input_file = 'Real_MTMR_pos_319.mat'
test_output_file = 'Real_MTMR_tor_319.mat'

model = LogNet(12, H, 6)
loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)
train_loader, valid_loader, output_scaler = load_train_data(train_input_file_list, train_output_file_list, valid_size, batch_size, device)
model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch)
test(model, test_input_file, test_output_file, output_scaler)

torch.save(model.state_dict(), pjoin('model','LogNet',model_file_name+'.pt'))
with open(pjoin('model','LogNet',model_file_name+'.pkl'), 'wb') as fid:
    cPickle.dump(output_scaler, fid)

