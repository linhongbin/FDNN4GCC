import torch
import _pickle as cPickle
from regularizeTool import EarlyStopping
from trainTool import train
from Net import *
from loadDataTool import load_train_data
from os.path import join
from evaluateTool import test

# path
pretrain_data_path = join("data", "CAD_sim_1e6")
train_data_path = join("data", "MTMR_real_8192")
test_data_path = join("data", "MTMR_real_319")

# config hyper-parameters
H = 1000  # number of hidden neurons
learning_rate = 0.01 # learning rate
max_training_epoch = 2000 # stop train when reach maximum training epoch
goal_loss = 1e-4 # stop train when reach goal loss
valid_ratio = 0.2 # ratio of validation data set over train and validate data
batch_size = 256 # batch size for mini-batch gradient descent
earlyStop_patience = 15 # epoch number of looking ahead


# load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#### pretrain using simulation data
train_loader, valid_loader, input_scaler, output_scaler, input_dim, output_dim = load_train_data(data_dir=pretrain_data_path,
                                                                                                 valid_ratio=valid_ratio,
                                                                                                 batch_size=batch_size,
                                                                                                 device=device)
# configure network and optimizer
# model = BPNet(input_dim, H, output_dim)
model = ReLuNet(6, [100, 100], 6).to(device)
loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)


# train model
model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch)



train_loader, valid_loader, input_scaler, output_scaler, input_dim, output_dim = load_train_data(data_dir=train_data_path,
                                                                                                 valid_ratio=valid_ratio,
                                                                                                 batch_size=batch_size,
                                                                                                 device=device)
# configure network and optimizer
# model = BPNet(input_dim, H, output_dim)
model = ReLuNet(6, [100, 100], 6).to(device)
loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)


# train model
model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch)
# test model
model = model.to('cpu')
test_loss, abs_rms_vec, rel_rms_vec = test(model, loss_fn, test_data_path, input_scaler, output_scaler, 'cpu')

# save model
# torch.save(model.state_dict(), pjoin('model','LogNet',model_file_name+'.pt'))
# with open(pjoin('model','LogNet',model_file_name+'.pkl'), 'wb') as fid:
#     cPickle.dump(output_scaler, fid)

