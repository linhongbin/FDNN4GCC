import torch
import _pickle as cPickle
from regularizeTool import EarlyStopping
from trainTool import pretrain, train
from Net import BPNet
from loadDataTool import load_train_data
from os.path import join
from evaluateTool import test

# path
pretrain_data_path = join("data", "CAD_sim_1e6")
train_data_path = join("data", "MTMR_real_8192")
test_data_path = join("data", "MTMR_real_319")

# config hyper-parameters
input_dim = 6
output_dim = 6
H = [20, 20]  # number of hidden neurons
learning_rate = 0.01 # learning rate
max_training_epoch = 2000 # stop train when reach maximum training epoch
goal_loss = 1e-3 # stop train when reach goal loss
valid_ratio = 0.2 # ratio of validation data set over train and validate data
batch_size = 256 # batch size for mini-batch gradient descent
earlyStop_patience = 10 # epoch number of looking ahead
# choose the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# configure network and optimizer
layer_list = []
io_list = [input_dim]
io_list.extend(H)
io_list.append(output_dim)

for i in range(len(io_list)-1):
    layer_list.append(torch.nn.Linear(io_list[i], io_list[i+1]))
    if i != len(io_list)-2:
        layer_list.append(torch.nn.ReLU())
model = torch.nn.Sequential(*layer_list).to(device)


# load data
pre_train_loader, pre_valid_loader, _, _, _, _ = load_train_data(data_dir=pretrain_data_path,
                                                              valid_ratio=valid_ratio,
                                                              batch_size=batch_size,
                                                              device=device)


model = pretrain(model, pre_train_loader, pre_valid_loader, learning_rate, earlyStop_patience, max_training_epoch)






train_loader, valid_loader, input_scaler, output_scaler, input_dim, output_dim = load_train_data(data_dir=train_data_path,
                                                                                                 valid_ratio=valid_ratio,
                                                                                                 batch_size=batch_size,
                                                                                                 device=device)



loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)
# train model
model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch)
# test model
test_loss, abs_rms_vec, rel_rms_vec = test(model, loss_fn, test_data_path, input_scaler, output_scaler, device)




# save model
# torch.save(model.state_dict(), pjoin('model','LogNet',model_file_name+'.pt'))
# with open(pjoin('model','LogNet',model_file_name+'.pkl'), 'wb') as fid:
#     cPickle.dump(output_scaler, fid)

