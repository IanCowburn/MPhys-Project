import uproot
import awkward as ak
import vector
from hist import Hist, axis
from matplotlib import pyplot as plt
import mplhep as hep
import numpy as np
import h5py
import pandas as pd
import torch
import requests
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

file_tt = uproot.open("tttt_NLO_523243_mc23a_fullsim.root")
tree = file_tt["reco"]
print(tree.keys())


# EventNumber = tree["eventNumber"].array()
# LeptonCharge = tree["lepton_charge"].array()

# New_array = tree.arrays(["eventNumber", "lepton_charge"])
# print(New_array)

# New_array_branch_names = {
#     "EN": "eventNumber",
#     "LC": "lepton_charge"
# }

# Selected_array = tree.arrays(New_array_branch_names.keys(), aliases = New_array_branch_names)

# print(Selected_array["EN"])
# print(Selected_array["LC"])

# print(len(Selected_array["LC"]))
# empty_mask = ak.num(Selected_array["LC"]) > 0
# lepton_charge_mask = ak.all(Selected_array["LC"] > 0, axis=1)
# Filtered_array = Selected_array[empty_mask & lepton_charge_mask]
# Number_of_events_left = len(Filtered_array["LC"])
# print(Number_of_events_left)

parton_vectors = vector.zip({
    "pt": tree["parton_top_pt"].array(),
    "eta": tree["parton_top_eta"].array(),
    "phi": tree["parton_top_phi"].array(),
    "mass": tree["parton_top_m"].array()
})

# print(parton_vectors.pt)
# print(ak.num(parton_vectors))



# parton_pt_hist = Hist(axis.Regular(50, 0, 1e6, name = "pt", label = "Parton $p_{T}$ [MeV]"))

# parton_pt_hist.fill(pt = parton_vectors[:,0].pt)

# fig, ax = plt.subplots(figsize=(8, 5))
# hep.histplot(parton_pt_hist, ax=ax)
# ax.set_xlabel("Parton $p_T$ [MeV]")
# ax.set_ylabel("Counts")
# ax.set_title("Parton $p_T$ Distribution")

# plt.show()



var_names = ["lepton_Id", "lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "HT_all_NOSYS", "lepton_pt_NOSYS", "jet_pt_NOSYS", "HT_fjets_NOSYS", "HT_jets_NOSYS"]

data_array = tree.arrays(var_names)
masking_array = tree.arrays(["nLeptons", "nJets"])
# print(ak.max(data_array["lepton_eta"]))
# print(ak.max(data_array["jet_eta"]))
# print(ak.max(data_array["lepton_phi"]))
# print(ak.max(data_array["jet_phi"]))

# print(ak.min(data_array["lepton_eta"]))
# print(ak.min(data_array["jet_eta"]))
# print(ak.min(data_array["lepton_phi"]))
# print(ak.min(data_array["jet_phi"]))

nLeptons_mask = masking_array["nLeptons"] <= 2
nJets_mask1 = masking_array["nJets"] <= 10
nJets_mask2 = masking_array["nJets"] >= 5
average = ak.mean(masking_array["nJets"])
standard_deviation = ak.std(masking_array["nJets"])
data_array = data_array[nLeptons_mask & nJets_mask1 & nJets_mask2]
parton_vectors = parton_vectors[nLeptons_mask & nJets_mask1 & nJets_mask2]

print(average)
print(standard_deviation)

combined_parton_system = parton_vectors[:,0] + parton_vectors[:,1] + parton_vectors[:,2] + parton_vectors[:,3]
print(combined_parton_system.m)

padding_size = []
for name in var_names:
    try:
        pad = int(ak.max(ak.num(data_array[name])))
    except Exception:
        pad = 1 
    padding_size.append(pad)

print(padding_size)

for i in range(len(var_names)):
    name = var_names[i]
    pad = padding_size[i]
    try:
        data_array[name] = ak.pad_none(data_array[name], pad, clip=True)
        data_array[name] = ak.fill_none(data_array[name], -99)
    except Exception:
        pass

print(data_array[0])


columns = []
for name in var_names:
    arr = ak.to_numpy(data_array[name])
    if arr.ndim == 2:
        for i in range(arr.shape[1]):
            if name == "lepton_phi" or name == "jet_phi":
                columns.append(np.cos(arr[:, i]))
            else:
                columns.append(arr[:, i])
    else:
        columns.append(arr)
data_array = np.column_stack(columns)

print(data_array.shape)
print(data_array[0])

class SimpleDNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleDNN, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
model = SimpleDNN(data_array.shape[1])


x_scaler = StandardScaler()
x_tensor = torch.tensor(x_scaler.fit_transform(data_array), dtype=torch.float32)
y_scaler = StandardScaler()
y_tensor = torch.tensor(y_scaler.fit_transform(ak.to_numpy(combined_parton_system.m).reshape(-1, 1)), dtype=torch.float32).reshape(-1, 1)

print(np.mean(x_tensor.numpy(), axis=0))
print(np.std(x_tensor.numpy(), axis=0))

loss_function = nn.MSELoss()
optimizer = torch.optim.Rprop(model.parameters())
N_epochs = 5000
for epoch in range(N_epochs):
    optimizer.zero_grad()
    predictions = model(x_tensor)
    loss = loss_function(predictions, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{N_epochs}, Loss: {loss.item():.4f}")

y_out = model(x_tensor).detach().numpy()
y_pred = y_scaler.inverse_transform(y_out)

print(ak.to_numpy(combined_parton_system.m))

dist = np.abs((ak.to_numpy(combined_parton_system.m).reshape(-1) - y_pred.reshape(-1)))

plt.hist2d(ak.to_numpy(combined_parton_system.m), y_pred.reshape(-1), bins=250, range=[[0, 6e6], [0, 6e6]], cmin=1)
plt.plot([0, 6e6], [0, 6e6], color='red', linestyle='--')
plt.colorbar(label = "Event Density")
plt.xlabel('True Parton System Mass [MeV]')
plt.ylabel('DNN Predicted Mass [MeV]')
plt.title('DNN Regression for Parton System Mass')
plt.savefig("MPhys4TopDNN_PartonMassRegression.png", dpi=600)
plt.show()

plt.hist(dist, bins=100, range=[0, 1e6])
plt.xlabel('Absolute Error [MeV]')
plt.ylabel('Event Density')
plt.title('DNN Regression Absolute Error')
plt.savefig("MPhys4TopDNN_PartonMassRegression_Error.png", dpi=600)
plt.show()

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
model.apply(reset_weights)

RMS = np.sqrt(np.mean(dist**2))
print(f"Root Mean Square Error (RMSE): {RMS} MeV")

