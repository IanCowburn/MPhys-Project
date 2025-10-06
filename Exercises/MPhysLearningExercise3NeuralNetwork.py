import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits import mplot3d
from torch import nn, optim

model_DNN = nn.Sequential(nn.Linear(1, 50),
                          nn.ReLU(),
                          nn.Linear(50, 1))


np.random.seed(0)
x = np.linspace(0, 10, 1000).reshape(-1, 1)
y = x**2 + 0.5 * x + np.random.randn(*x.shape)

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

loss_function = nn.MSELoss()
optimizer = optim.Rprop(model_DNN.parameters())

N_epochs = 5000

for epoch in range(N_epochs):
    optimizer.zero_grad()
    predictions = model_DNN(x_tensor)
    loss = loss_function(predictions, y_tensor)
    loss.backward()
    optimizer.step()

y_out = model_DNN(x_tensor)
y_pred = y_out.detach()

plt.scatter(x, y, color='blue', label='Data Points', marker='.')
plt.plot(x, y_pred.numpy(), color='red', label='DNN Prediction')
plt.xlabel('Input')
plt.ylabel('Target')
plt.title('DNN Regression Example')
plt.legend()
plt.show()