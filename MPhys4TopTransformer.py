import uproot
import awkward as ak
import vector
from matplotlib import pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

file_tt = uproot.open("tttt_NLO_523243_mc23a_fullsim.root")
tree = file_tt["reco"]
print(tree.keys())

parton_vectors = vector.zip({
    "pt": tree["parton_top_pt"].array(),
    "eta": tree["parton_top_eta"].array(),
    "phi": tree["parton_top_phi"].array(),
    "mass": tree["parton_top_m"].array()
})

lepton_mask_size = 2
jet_mask_size = 12

var_names = ["lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "lepton_pt_NOSYS", "jet_pt_NOSYS"]
data_array = tree.arrays(var_names)
masking_array = tree.arrays(["jet_eta", "lepton_eta"])
nLeptons_mask = ak.num(masking_array["lepton_eta"]) <= lepton_mask_size
nJets_mask = ak.num(masking_array["jet_eta"]) <= jet_mask_size
data_array = data_array[nLeptons_mask & nJets_mask]
parton_vectors = parton_vectors[nLeptons_mask & nJets_mask]
combined_parton_system = parton_vectors[:,0] + parton_vectors[:,1] + parton_vectors[:,2] + parton_vectors[:,3]
padding_size = []


print(ak.type(data_array))

for name in var_names:
    pad = int(ak.max(ak.num(data_array[name])))
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

lepton_arrays = []
jet_arrays = []

for name in var_names:
    if "lepton" in name:
        lepton_arrays.append(data_array[name])
    elif "jet" in name:
        jet_arrays.append(data_array[name])

lepton_arrays = ak.to_numpy(lepton_arrays)
jet_arrays = ak.to_numpy(jet_arrays)

print(lepton_arrays.shape)
print(jet_arrays.shape)

data_array = np.concatenate([lepton_arrays, jet_arrays], axis = 2)
print(data_array.shape)
data_array = np.transpose(data_array, (1,2,0))
print(data_array.shape)

class TransformerEncoder(nn.Module):
    """
    Super simple transformer encoder
    """
    def __init__(self, d_model, n_heads, num_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
    def forward(self, x, mask=None):
        return self.encoder(x,src_key_padding_mask=mask)
    
class Embedder(nn.Module):
    """
    Simple embedding layer to upscale input features to d_model dimensions.
    The masked 
    Args:
        in_features (int): Number of input features per token.
        d_model (int): Desired output feature dimension.
    Returns:
        torch.Tensor: Embedded output of shape (batch, tokens, d_model).
    """
    def __init__(self, in_features, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features, d_model)
    def forward(self, x, src_key_padding_mask):
        # 1. Apply linear transformation to upscale input features. 
        # x shape: (batch, tokens, features)
        embedded_x = self.linear(x)
        # 2. Use the mask to zero out the embedded vectors for padded tokens.
        mask = src_key_padding_mask.unsqueeze(-1)
        # x shape: (batch, tokens, d_model) with padded tokens set to zero
        embedded_x = embedded_x.masked_fill(mask, 0.0)
        return embedded_x
    

    # --- Regression Model ---
class TransformerRegressor(nn.Module):
    """
    Transformer model for regression (predicts invariant mass).
    """
    def __init__(self, input_dim, embed_dim, n_heads, num_layers):
        super().__init__()
        self.embedder = Embedder(input_dim, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers)
        self.regressor = nn.Linear(embed_dim, 1)
    def forward(self, x):
        src_key_padding_mask = (x == -99.0).all(dim=-1)
        x = self.embedder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.transformer(x, mask=src_key_padding_mask)
        # Take mean over sequence (tokens) dimension for regression
        x = x.mean(dim=1)
        out = self.regressor(x)
        return out.squeeze(-1)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- Data Preparation ---
# Features: data_array (already numpy)
# Target: invariant mass from combined_parton_system
X = data_array.astype(np.float32)
y = ak.to_numpy(combined_parton_system.mass).astype(np.float32)
print(f"X shape before mask: {X.shape}")
print(f"y shape before mask: {y.shape}")

# Remove any rows with -99 padding in all features (optional, for clean training)
valid_mask = ~(X == -99.0).all(axis=(1, 2))
print(f"valid_mask shape: {valid_mask.shape}, keeping {valid_mask.sum()} samples")

X = X[valid_mask]
y = y[valid_mask]

print(f"X shape after mask: {X.shape}")
print(f"y shape after mask: {y.shape}")


# Standardize features
scaler_X = StandardScaler()
E, F, T = X.shape
print(E, F, T)
X_for_scaling = np.transpose(X, (0, 2, 1))
print(X_for_scaling.shape)
X_for_scaling = X_for_scaling.reshape(-1, F)
X_scaled = scaler_X.fit_transform(X_for_scaling)
X_scaled = X_scaled.reshape(E, T, F)
print(X_scaled.shape)
X_scaled = np.transpose(X_scaled, (0, 2, 1))
print(X_scaled.shape)


# Standardize target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Convert to torch tensors
X_tensor = torch.from_numpy(X_scaled)
y_tensor = torch.from_numpy(y_scaled)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Model, Loss, Optimizer ---
input_dim = X_train.shape[2]
embed_dim = 128
n_heads = 4
num_layers = 3
model = TransformerRegressor(input_dim, embed_dim, n_heads, num_layers)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# --- Training Loop ---
epochs = 50
batch_size = 128
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))
    train_loss = 0.0
    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train[idx]
        yb = y_train[idx]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= X_train.size(0)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

# --- Prediction ---
model.eval()
with torch.no_grad():
    X_test_seq = X_test.unsqueeze(1)  # (batch, seq_len=1, features)
    y_pred_scaled = model(X_test_seq).cpu().numpy()
    y_true_scaled = y_test.cpu().numpy()

# Inverse transform to get physical invariant mass values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

# --- 2D Histogram Plot ---
plt.figure(figsize=(8,6))
plt.hist2d(y_true, y_pred, bins=250, range=[[0, 6e6], [0, 6e6]], cmap='viridis', cmin=1)
plt.xlabel('Expected Invariant Mass')
plt.ylabel('Predicted Invariant Mass')
plt.title('2D Histogram: Expected vs Predicted Invariant Mass')
plt.colorbar(label='Counts')
plt.savefig('transformer_regression_2d_histogram.png', dpi=300)
plt.show()