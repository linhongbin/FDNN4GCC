import torch
from Net import *
device = torch.device('cpu')


N, D_in, D_out = 64, 3, 3


x = torch.randn(N, D_in, device=device)
y = x + 0.5*x.pow(2) + 0.1 * x.pow(3) +0.6

learning_rate = 2e-1
model = PolNet(D_in,5)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



