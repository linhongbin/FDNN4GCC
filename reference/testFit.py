from os.path import join as pjoin
import scipy.io as sio
import torch
from sklearn import preprocessing
import numpy as np
import _pickle as cPickle
from sklearn.metrics import mean_squared_error

input_mat = sio.loadmat(pjoin('CAD_sim_1e6','Real_MTMR_pos_4096.mat'))['input_mat']
output_mat = sio.loadmat(pjoin('CAD_sim_1e6','Real_MTMR_tor_4096.mat'))['output_mat']

device = torch.device('cpu')
H = 1000

input_mat = input_mat.T
output_mat = output_mat.T
input_mat = input_mat[:, :-1]
output_mat = output_mat[:, :-1]
input_scaler = preprocessing.StandardScaler().fit(input_mat)
output_scaler = preprocessing.StandardScaler().fit(output_mat)
input_mat = input_scaler.transform(input_mat)
output_mat = output_scaler.transform(output_mat)



x = torch.from_numpy(input_mat).to(device)
y = torch.from_numpy(output_mat).to(device)
x = x.float()
y = y.float()

(N_in, D_in) = x.shape
(N_out, D_out) = y.shape
assert N_in == N_out

model = torch.nn.Sequential(
            torch.nn.Linear(6, 400),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(400, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 6)).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for t in range(4000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    loss_val = loss.item()
    print('Epoch', t, ': Loss is ', loss_val)

    if loss_val<1e-3:
        break


# test model
test_input_mat = sio.loadmat(pjoin('CAD_sim_1e6','MTMR_28002_traj_test_10_pos.mat'))['input_mat']
test_output_mat = sio.loadmat(pjoin('CAD_sim_1e6','MTMR_28002_traj_test_10_tor.mat'))['output_mat']
test_input_mat = test_input_mat.T
test_input_mat = test_input_mat[:, :-1]
test_input_mat = input_scaler.transform(test_input_mat)
test_input_mat = torch.from_numpy(test_input_mat).to(device)
test_input_mat = test_input_mat.float()
y_pred = model(test_input_mat)
test_output_mat_hat = output_scaler.inverse_transform(y_pred.data.numpy())
test_output_mat = test_output_mat.T
test_output_mat = test_output_mat[:, :-1]
e_mat = np.sqrt(np.divide(np.sum(np.square(test_output_mat_hat - test_output_mat), axis=0),
                  np.sum(np.square(test_output_mat_hat), axis=0)))

print(e_mat)
model_file_name = './model/LeakReLUFitReal4096'

torch.save(model.state_dict(), './model/LeakReLUFitReal4096'+'.pt')


with open('model/LeakReLUFitReal4096'+'.pkl', 'wb') as fid:
    cPickle.dump(input_scaler, fid)
    cPickle.dump(output_scaler, fid)
