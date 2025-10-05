import pandas as pd
import matplotlib.pyplot as plt
import torch
from mpl_toolkits import mplot3d
from torch import nn, optim

input_penguins_df = pd.read_csv('Exercises/penguins.csv')
penguins_df = input_penguins_df.dropna(inplace=False)

input_data1 = torch.tensor(penguins_df["flipper_length_mm"].values, dtype=torch.float32).reshape(-1,1)
input_data2 = torch.tensor(penguins_df["bill_length_mm"].values,    dtype=torch.float32).reshape(-1,1)

input_data2D = torch.tensor(penguins_df[["flipper_length_mm", "bill_length_mm"]].values, dtype=torch.float32)

target     = torch.tensor(penguins_df["body_mass_g"].values, dtype=torch.float32).reshape(-1,1)

model = nn.Linear(2, 1)

print(model)
print(list(model.parameters()))

loss_function = nn.MSELoss()
optimizer = optim.Rprop(model.parameters())

# keep track of the loss every epoch. This is only for visualisation
losses = []

N_epochs = 5000

for epoch in range(N_epochs):
    # tell the optimizer to begin an optimization step
    optimizer.zero_grad()

    # use the model as a prediction function: features → prediction
    predictions = model(input_data2D)

    if epoch == 1000:
        print(predictions)

    # compute the loss (χ²) between these predictions and the intended targets
    loss = loss_function(predictions, target)

    # tell the loss function and optimizer to end an optimization step
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

y_out = model(input_data2D)
y_pred = y_out.detach()

def r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r_squared_value = r_squared(target, y_pred)
print(r_squared_value.item())

ax = plt.axes(projection='3d')

ax.scatter3D(input_data2D[:,0].reshape(-1,1), input_data2D[:,1].reshape(-1,1), target.reshape(-1,1), color='blue', label='Data Points')
ax.scatter3D(input_data2D[:,0].reshape(-1,1), input_data2D[:,1].reshape(-1,1), y_pred, color='red', label='Linear Regression Plane')
ax.set_xlabel('Flipper Length (mm)')
ax.set_ylabel('Bill Length (mm)')
ax.set_zlabel('Body Mass (g)')
ax.set_title('3D Linear Regression Example')
plt.legend()
plt.show()