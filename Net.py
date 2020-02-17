import torch

class PolNet(torch.nn.Module):
    def __init__(self, D_in, pol_dim,  device='cpu'):
        super(PolNet, self).__init__()
        self.D_in = D_in
        self.pol_dim = pol_dim
        self.device = device
        self.linears = torch.nn.ModuleList([torch.nn.Linear(pol_dim+1, 1) for i in range(D_in)])
    def forward(self, x):
        y = torch.ones([x.shape[0], self.D_in], dtype=torch.float32, device=self.device)
        for j in range(self.D_in):
            feature = torch.ones([x.shape[0], self.pol_dim+1], dtype=torch.float32, device=self.device)
            for i in range(self.pol_dim):
                feature[:,i+1] = x[:,j].pow(i+1).squeeze()
            linear = self.linears[j]
            y_pred = linear(feature)
            y[:,j] = y_pred.squeeze()
        return y

class VanillaNet(torch.nn.Module):
    def __init__(self, base_net, addition_net):
        super(VanillaNet, self).__init__()
        self.base_net = base_net
        self.addition_net = addition_net
    def forward(self, x):
        y1 = self.base_net(x)
        y = self.addition_net(x) + y1
        return y

# class LagrangeNet(torch.nn.Module):
#     def __init__(self, base_model, delta_q, w_vec, device='cpu'):
#         super(LagrangeNet, self).__init__()
#         self.base_model = base_model
#         self.delta_q = delta_q
#         self.w_vec = w_vec
#         self.device = device
#     def forward(self, feature):
#         target_hat = torch.zeros(feature.shape, dtype=torch.float32, device=self.device)
#         base_hat = self.base_model(feature)
#         base_hat.require_grad = False
#         for j in range(feature.shape[1]):
#             feature_d = feature.clone()
#             feature_d[:, j] = feature_d[:, j].clone() + torch.ones(feature_d[:, j].shape,
#                                                                    device=self.device).float() * self.delta_q
#             target_hat[:, j] = self.base_model(feature_d).squeeze() - base_hat.squeeze()
#         target_hat = target_hat / self.delta_q
#         target_hat = torch.mul(target_hat, self.w_vec)
#         return target_hat

class LagrangeNet(torch.nn.Module):
    def __init__(self, base_model, device='cpu'):
        super(LagrangeNet, self).__init__()
        self.add_module("base_model", base_model)
        self.device = device
    def forward(self, feature):
        feature.requires_grad_(True)
        base_hat = self.base_model(feature)
        target_hat = torch.autograd.grad(outputs=base_hat, inputs=feature, grad_outputs=torch.ones(base_hat.shape), retain_graph=True,
                                      create_graph=True)[0]
        return target_hat

# def Lagrange_Net(model, feature, delta_q, w_vec, device='cpu'):
#     target_hat = torch.zeros(feature.shape, dtype=torch.float32, device=device)
#     base_hat = model(feature)
#     base_hat.require_grad = False
#     for j in range(feature.shape[1]):
#         feature_d = feature.clone()
#         feature_d[:, j] = feature_d[:, j].clone() + torch.ones(feature_d[:, j].shape, device=device).float() * delta_q
#         target_hat[:, j] = model(feature_d).squeeze()-base_hat.squeeze()
#     target_hat = target_hat / delta_q
#     target_hat = torch.mul(target_hat, w_vec)
#     return target_hat


# create Net architecture
class LogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LogNet, self).__init__()
        H1 = 100
        H2 = 50
        self._relu = torch.nn.ReLU()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._middle_linear = torch.nn.Linear(H, H)
        self._output_linear = torch.nn.Linear(H, D_out)
        self._epsilon = 1

    def forward(self, x):
        h = x
        h = self._input_linear(h).clamp(self._epsilon)
        h = torch.log(h)
        h = self._middle_linear(h)
        h = torch.exp(h)
        y_pred = self._output_linear(h)
        return y_pred

class SinLogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SinLogNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)
        self._epsilon = 0.003
        self.bias = torch.nn.Parameter(torch.randn(1, D_in))

    def forward(self, x):
        h = x + self.bias
        h1 = torch.sin(h)
        h = h.clamp(self._epsilon)
        h = torch.log(h)
        h = self._input_linear(h)
        h = torch.exp(h)
        y_pred = self._output_linear(h)
        return y_pred

# create Net architecture
class BPNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(BPNet, self).__init__()
        self._tanh = torch.nn.Tanh()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = self._tanh(h)
        y_pred = self._output_linear(h)
        return y_pred

class ReLuNet(torch.nn.Module):
    def __init__(self, D_in, H_list, D_out):
        super(ReLuNet, self).__init__()
        self.relu = torch.nn.ReLU()
        H_list = [D_in] + H_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(H_list[i], H_list[i+1]) for i in range(len(H_list)-1)])
        self.output_layer = torch.nn.Linear(H_list[-1], D_out)
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
        x = self.output_layer(x)
        return x

class SinNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SinNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._middle_linear = torch.nn.Linear(H, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = torch.sin(h)
        y_pred = self._output_linear(h)
        return y_pred

class Multi_SinNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Multi_SinNet, self).__init__()
        self.sin_layers = torch.nn.ModuleList([SinNet(i+1, H, D_out) for i in range(D_in)])
        self.D_in = D_in
        self.D_out = D_out


    def forward(self, x):
        y_pred = torch.zeros([x.shape[0], self.D_out], dtype=torch.float32, device='cpu')
        for i in range(self.D_in):
            layer = self.sin_layers[i]
            y_pred = y_pred + layer(x[:,:i+1])
        return y_pred


class SigmoidNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SigmoidNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = torch.sigmoid(h)
        y_pred = self._output_linear(h)
        return y_pred


class KDNet_Parallel(torch.nn.Module):
    def __init__(self, DOF, K_VecNum, Com_LayerList, K_LayerList, D_LayerList):
        super(KDNet_Parallel, self).__init__()
        self.act_func = torch.nn.ReLU()
        numList = [DOF]
        numList.extend(Com_LayerList)
        self.Com_Linears = torch.nn.ModuleList([torch.nn.Linear(numList[i], numList[i + 1]) for i in range(len(numList) - 1)])

        numList = [Com_LayerList[-1]]
        numList.extend(K_LayerList)
        numList.append(K_VecNum)
        self.K_Linears = torch.nn.ModuleList([torch.nn.Linear(numList[i], numList[i + 1]) for i in range(len(numList) - 1)])

        numList = [Com_LayerList[-1]]
        numList.extend(D_LayerList)
        numList.append(DOF)
        self.D_Linears = torch.nn.ModuleList([torch.nn.Linear(numList[i], numList[i + 1]) for i in range(len(numList) - 1)])


    def forward(self, x):
        for linear in self.Com_Linears:
            x = linear(x)
            x = self.act_func(x)


        k = x
        for i in range(len(self.K_Linears)):
            k = self.K_Linears[i](k)
            if i is not len(self.K_Linears)-1:
                k = self.act_func(k)

        d = x
        for i in range(len(self.D_Linears)):
            d = self.D_Linears[i](d)
            if i is not len(self.D_Linears)-1:
                d = self.act_func(d)

        return d, k


class KDNet_Serial(torch.nn.Module):
    def __init__(self, DOF, K_LayerList, D_LayerList, K_VecNum):
        super(KDNet_Serial, self).__init__()
        self.act_func = torch.nn.ReLU()

        numList = [DOF]
        numList.extend(K_LayerList)
        numList.append(K_VecNum)
        self.K_Linears = torch.nn.ModuleList([torch.nn.Linear(numList[i], numList[i + 1]) for i in range(len(numList) - 1)])

        numList = [K_VecNum]
        numList.extend(D_LayerList)
        numList.append(DOF)
        self.D_Linears = torch.nn.ModuleList([torch.nn.Linear(numList[i], numList[i + 1]) for i in range(len(numList) - 1)])


    def forward(self, x):
        k = x
        for i in range(len(self.K_Linears)):
            k = self.K_Linears[i](k)
            if i is len(self.K_Linears)-1:
                k_out = k
            k = self.act_func(k)

        d = k
        for i in range(len(self.D_Linears)):
            d = self.D_Linears[i](d)
            if i is not len(self.D_Linears)-1:
                d = self.act_func(d)
        d_out = d
        return d_out, k_out

    def freeze_KLayers(self):
        for linear in self.K_Linears:
            for param in linear.parameters():
                param.requires_grad = False

    def reset_DLayers(self):
        for linear in self.D_Linears:
            torch.nn.init.xavier_uniform_(linear.weight)
            linear.bias.data.fill_(0.01)


    def reset_KLastLayer(self):
        torch.nn.init.xavier_uniform_(self.K_Linears[-1].weight)
        self.K_Linears[-1].bias.data.fill_(0.01)

    def reset_after_pretrain(self):
        self.reset_DLayers()
        self.reset_KLastLayer()
        for i in range(len(self.K_Linears)):
            if i is not len(self.K_Linears)-1:
                for param in self.K_Linears[i].parameters():
                    param.requires_grad = False

    def unfreeze_KLayers(self):
        for linear in self.K_Linears:
            for param in linear.parameters():
                param.requires_grad = True


class SinInput_ReLUNet(torch.nn.Module):
    def __init__(self, D_in, H_list, D_out):
        super(SinInput_ReLUNet, self).__init__()
        self.relu = torch.nn.ReLU()
        H_list = [2*D_in] + H_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(H_list[i], H_list[i+1]) for i in range(len(H_list)-1)])
        self.output_layer = torch.nn.Linear(H_list[-1], D_out)
    def forward(self, x):
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        x = torch.cat((x_sin, x_cos), 1)
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
        x = self.output_layer(x)
        return x


class DualNets_UDirection(torch.nn.Module):
    def __init__(self, pos_net, neg_net, dof,device='cpu'):
        super(DualNets_UDirection, self).__init__()
        self.pos_net = pos_net
        self.neg_net = neg_net
        self.dof = dof
        self.device = device
        self.input_mean = None
        self.input_std = None
        self.output_mean =None
        self.output_std = None

    def forward(self, x):
        z = x[:,:self.dof*2]
        delta_q = x[:,self.dof*2:]
        tor_pos = self.pos_net(z)
        tor_neg = self.neg_net(z)
        u_mat = torch.zeros(delta_q.shape).to(self.device)
        u_mat[delta_q>0]=1
        tor = tor_pos * u_mat + tor_neg * (1-u_mat)
        return tor

    def set_normalized_param(self,input_mean, input_std, output_mean, output_std):
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean =output_mean
        self.output_std = output_std


    def predict_NP(self, input_mat):
        D = self.dof
        for i in range(D * 2):
            input_mat[:,i] = (input_mat[:,i] - self.input_mean[i]) / self.input_std[i]

        input = torch.from_numpy(input_mat).to('cpu').float()
        output = self.forward(input)
        output_mat = output.detach().numpy()

        for i in range(D):
            output_mat[:,i] = (output_mat[:,i] * self.output_std[i]) + self.output_mean[i]


        return output_mat


