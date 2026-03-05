import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import gridspec
from FourTopDualRegressorModel import TransformerEncoder, LeptonEmbedder, JetEmbedder, METEmbedder, NumbersEmbedder, HTEmbedder

X = np.load("dual_cache_X.npy")
y = np.load("dual_cache_y.npy")
pad_mask_np = np.load("dual_cache_pad_mask.npy")

class TransformerScaling:
    def __init__(self, X, pad_mask_np, y,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht):
        self.X = X
        self.pad_mask_np = pad_mask_np
        self.y = y
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_numbers = in_features_numbers
        self.in_features_ht = in_features_ht
    def scale_y(self, y_train, y_valid, y_test):
        # Use z-score scaling only (no log transform).
        def to_numpy(arr):
            return arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)

        y_train_np = to_numpy(y_train)
        y_valid_np = to_numpy(y_valid)
        y_test_np = to_numpy(y_test)

        scaler = StandardScaler()

        y_train_proc = y_train_np
        y_valid_proc = y_valid_np
        y_test_proc = y_test_np

        if y_train_proc.ndim == 1:
            y_train_proc = y_train_proc.reshape(-1, 1)
            y_valid_proc = y_valid_proc.reshape(-1, 1)
            y_test_proc = y_test_proc.reshape(-1, 1)

        y_train_scaled = scaler.fit_transform(y_train_proc)
        y_valid_scaled = scaler.transform(y_valid_proc)
        y_test_scaled = scaler.transform(y_test_proc)

        y_train_scaled = torch.from_numpy(y_train_scaled).float()
        y_valid_scaled = torch.from_numpy(y_valid_scaled).float()
        y_test_scaled = torch.from_numpy(y_test_scaled).float()

        return y_train_scaled, y_valid_scaled, y_test_scaled, scaler
    def prepare_data(self, X, y_scaled, pad_mask_np):
        valid_tokens = ~pad_mask_np  # numpy (E,T)
        # Make everything tensors before split so train_test_split returns tensors consistently
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y_scaled).float()
        pad_t = torch.from_numpy(pad_mask_np).bool()
        valid_t = torch.from_numpy(valid_tokens).bool()
        X_train, X_test, y_train, y_test, mask_train, mask_test, valid_train, valid_test = train_test_split(
            X_t, y_t, pad_t, valid_t, test_size=0.2, random_state=42
        )
        X_valid, X_test, y_valid, y_test, mask_valid, mask_test, valid_valid, valid_test = train_test_split(
            X_test, y_test, mask_test, valid_test, test_size=0.5, random_state=42
        )
        return (X_train, X_valid, X_test,
                y_train, y_valid, y_test,
                mask_train, mask_valid, mask_test,
                valid_train, valid_valid, valid_test)
    def scale_X(self, X_tensor, pad_mask_tensor, scalers):
        X = X_tensor.numpy().copy()
        pad_mask = pad_mask_tensor.numpy()

        lepton_slice = slice(0, self.lepton_mask_size)
        jet_slice = slice(self.lepton_mask_size, self.lepton_mask_size + self.jet_mask_size)
        met_idx = self.lepton_mask_size + self.jet_mask_size
        numbers_idx = met_idx + 1
        ht_idx = numbers_idx + 1

        for token_slice in (lepton_slice, jet_slice):
            group = X[:, token_slice, :4]
            valid_rows = (~pad_mask[:, token_slice]).reshape(-1)
            flat_group = group.reshape(-1, 4)
            if np.any(valid_rows):
                flat_group[valid_rows] = scalers["lepjet"].transform(flat_group[valid_rows])
            X[:, token_slice, :4] = flat_group.reshape(group.shape)

        X[:, met_idx, :4] = scalers["met"].transform(X[:, met_idx, :4])
        X[:, numbers_idx, :5] = scalers["numbers"].transform(X[:, numbers_idx, :5])
        X[:, ht_idx, :3] = scalers["ht"].transform(X[:, ht_idx, :3])

        return torch.from_numpy(X.astype(np.float32)).float()
    def __call__(self):
        (X_train, X_valid, X_test,
         y_train, y_valid, y_test,
         mask_train, mask_valid, mask_test,
         valid_train, valid_valid, valid_test) = self.prepare_data(self.X, self.y, self.pad_mask_np)
        y_train, y_valid, y_test, scaler_y = self.scale_y(y_train, y_valid, y_test)
        
        X_train_np = X_train.numpy().copy()
        mask_train_np = mask_train.numpy()

        lepton_slice = slice(0, self.lepton_mask_size)
        jet_slice = slice(self.lepton_mask_size, self.lepton_mask_size + self.jet_mask_size)
        met_idx = self.lepton_mask_size + self.jet_mask_size
        numbers_idx = met_idx + 1
        ht_idx = numbers_idx + 1

        lep_kin_flat = X_train_np[:, lepton_slice, :4].reshape(-1, 4)
        jet_kin_flat = X_train_np[:, jet_slice, :4].reshape(-1, 4)
        lep_valid_flat = (~mask_train_np[:, lepton_slice]).reshape(-1)
        jet_valid_flat = (~mask_train_np[:, jet_slice]).reshape(-1)
        lepjet_fit_data = np.concatenate([
            lep_kin_flat[lep_valid_flat],
            jet_kin_flat[jet_valid_flat]
        ], axis=0)

        scalers = {
            "lepjet": StandardScaler().fit(lepjet_fit_data),
            "met": StandardScaler().fit(X_train_np[:, met_idx, :4]),
            "numbers": StandardScaler().fit(X_train_np[:, numbers_idx, :5]),
            "ht": StandardScaler().fit(X_train_np[:, ht_idx, :3])
        }

        X_train = self.scale_X(X_train, mask_train, scalers)
        X_valid = self.scale_X(X_valid, mask_valid, scalers)
        X_test  = self.scale_X(X_test,  mask_test,  scalers)
        print(f"Training set:   {X_train.shape[0]} events")
        print(f"Validation set: {X_valid.shape[0]} events")
        print(f"Test set:       {X_test.shape[0]} events")
        return (X_train, X_valid, X_test,
                y_train, y_valid, y_test,
                mask_train, mask_valid, mask_test,
                scaler_y)

scaler = TransformerScaling(X, pad_mask_np, y,
                            lepton_mask_size=2, jet_mask_size=12,
                            in_features_leptons=5, in_features_jets=5,
                            in_features_met=4, in_features_numbers=5, in_features_ht=3)
(X_train, X_valid, X_test,
 y_train, y_valid, y_test,
 mask_train, mask_valid, mask_test,
 scaler_y) = scaler()

class TransformerContextEmbedder(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht):
        super().__init__()
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_numbers = in_features_numbers
        self.in_features_ht = in_features_ht
        self.lepton_embedder = LeptonEmbedder(in_features_leptons, embed_dim)
        self.jet_embedder = JetEmbedder(in_features_jets, embed_dim)
        self.met_embedder = METEmbedder(in_features_met, embed_dim)
        self.numbers_embedder = NumbersEmbedder(in_features_numbers, embed_dim)
        self.ht_embedder = HTEmbedder(in_features_ht, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers)

    def forward(self, x, pad_mask):
        # x: (B, T, F), pad_mask: (B, T) with True for pads
        leptons = x[:, :self.lepton_mask_size, :self.in_features_leptons]
        jets = x[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size, :self.in_features_jets]
        met = x[:, self.lepton_mask_size + self.jet_mask_size:self.lepton_mask_size + self.jet_mask_size + 1, :self.in_features_met]
        numbers = x[:, self.lepton_mask_size + self.jet_mask_size + 1:self.lepton_mask_size + self.jet_mask_size + 2, :self.in_features_numbers]
        ht_features = x[:, self.lepton_mask_size + self.jet_mask_size + 2:self.lepton_mask_size + self.jet_mask_size + 3, :self.in_features_ht]
        lepton_mask = pad_mask[:, :self.lepton_mask_size]
        jet_mask = pad_mask[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size]
        lepton_embedded = self.lepton_embedder(leptons, lepton_mask)
        jet_embedded = self.jet_embedder(jets, jet_mask)
        met_embedded = self.met_embedder(met)
        numbers_embedded = self.numbers_embedder(numbers)
        ht_embedded = self.ht_embedder(ht_features)
        embedded = torch.cat((lepton_embedded, jet_embedded, met_embedded, numbers_embedded, ht_embedded), dim=1)
        h = self.transformer(embedded, mask=pad_mask)
        valid = (~pad_mask).float()
        pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled
    
def sample_base(batch_size, dim, device):
    return torch.randn(batch_size, dim, device=device)

class ConditionalVelocityNet(nn.Module):
    def __init__(self, Ninputs, Ncontext, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Ninputs + 1 + Ncontext, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, Ninputs)
        )

    def forward(self, x, t, c):
        inp = torch.cat([x, t, c], dim=1)
        return self.net(inp)

def conditional_flow_matching_loss(VelocityNet, Embedder, train_X_batch, train_y_batch, pad_mask_batch):
    
    # Sample batch
    batch_size = train_X_batch.shape[0]
    device = train_X_batch.device
    x1 = train_y_batch
    if x1.dim() == 1:
        x1 = x1.unsqueeze(1)
    x0 = sample_base(batch_size, x1.shape[1], device)
    
    # Use context embeddor to get context from train_X_batch
    c = Embedder(train_X_batch, pad_mask_batch)
    
    # Conditional target sampling
    x1 = x1.to(device)
    
    # Sample time t
    t = torch.rand(batch_size, 1, device=device)
    
    # Interpolate between x0 and x1
    xt = (1 - t) * x0 + t * x1

    v_pred = VelocityNet(xt, t, c)
    v_target = x1 - x0

    return ((v_pred - v_target) ** 2).mean()

def plot_loss_curves(train_losses, val_losses, outpath="Flow_Matching_loss_curves.png"):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Flow Matching Training Curves")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

device = torch.device("cuda")
N = X_train.shape[0]
batch_size = 256
num_epochs = 100
target_dim = y_train.shape[1] if y_train.dim() == 2 else 1

VelNet = ConditionalVelocityNet(Ninputs=target_dim, Ncontext=64, hidden=128).to(device)
Embedder = TransformerContextEmbedder(
    embed_dim=64,
    n_heads=8,
    num_layers=6,
    lepton_mask_size=2,
    jet_mask_size=12,
    in_features_leptons=5,
    in_features_jets=5,
    in_features_met=4,
    in_features_numbers=5,
    in_features_ht=3,
).to(device)

optimizer = torch.optim.Adam(
    list(VelNet.parameters()) + list(Embedder.parameters()),
    lr=1e-3
)

patience = 5
min_epochs = 0
best_val_rounded = None
epochs_no_improve = 0

losses = []
val_losses = []
for epoch in range(num_epochs):
    perm = torch.randperm(N)

    total_loss = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]

        X_batch = X_train[idx].to(device)
        y_batch = y_train[idx].to(device)
        mask_batch = mask_train[idx].to(device)

        optimizer.zero_grad()

        loss = conditional_flow_matching_loss(
            VelNet,
            Embedder,
            X_batch,
            y_batch,
            mask_batch,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)

    avg_loss = total_loss / N
    losses.append(avg_loss)
    # Validation loss
    VelNet.eval()
    Embedder.eval()
    with torch.no_grad():
        val_total = 0.0
        val_count = 0
        for i in range(0, X_valid.shape[0], batch_size):
            Xv = X_valid[i:i+batch_size].to(device)
            yv = y_valid[i:i+batch_size].to(device)
            mv = mask_valid[i:i+batch_size].to(device)
            vloss = conditional_flow_matching_loss(VelNet, Embedder, Xv, yv, mv)
            val_total += vloss.item() * Xv.shape[0]
            val_count += Xv.shape[0]
        val_loss = val_total / max(val_count, 1)
        val_losses.append(val_loss)
    VelNet.train()
    Embedder.train()

    val_rounded = round(val_loss, 4)
    if best_val_rounded is None or val_rounded < best_val_rounded:
        best_val_rounded = val_rounded
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Val: {val_loss:.6f}")
    if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch:03d}: val rounded to 4dp did not improve for {patience} epochs.")
        break

plot_loss_curves(losses, val_losses)
print("Saved loss curves to Flow_Matching_loss_curves.png")

@torch.no_grad()
def sample_flow_single_batch(model, embedder, X_test, pad_mask, target_dim, n_steps=250, device="cuda"):
    B = X_test.shape[0]

    # Condition
    c = embedder(X_test.to(device), pad_mask.to(device))

    # Initial noise
    x = torch.randn(B, target_dim, device=device)

    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B, 1), i / n_steps, device=device)
        v = model(x, t, c)
        x = x + dt * v

    return x


@torch.no_grad()
def sample_flow(
    model,
    embedder,
    X_test,
    pad_mask,
    target_dim,
    n_steps=250,
    batch_size=None,
    device="cuda",
):
    B = X_test.shape[0]
    if batch_size is None or B <= batch_size:
        return sample_flow_single_batch(model, embedder, X_test, pad_mask, target_dim, n_steps=n_steps, device=device)

    outputs = []
    for start in range(0, B, batch_size):
        end = start + batch_size
        xb = X_test[start:end]
        mb = pad_mask[start:end]
        out = sample_flow_single_batch(model, embedder, xb, mb, target_dim, n_steps=n_steps, device=device)
        outputs.append(out)
    return torch.cat(outputs, dim=0)

@torch.no_grad()
def sample_flow_repeated(
    model,
    embedder,
    X_test,
    pad_mask,
    target_dim,
    n_steps=250,
    n_samples=1,
    batch_size=None,
    device="cuda",
):
    all_samples = []
    for _ in range(n_samples):
        samples = sample_flow(
            model,
            embedder,
            X_test,
            pad_mask,
            target_dim,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
        )
        all_samples.append(samples)
    return all_samples

x_pred_single = sample_flow(VelNet, Embedder, X_test, mask_test, target_dim, n_steps=250, batch_size=512)

x_pred = sample_flow_repeated(
    VelNet,
    Embedder,
    X_test,
    mask_test,
    target_dim,
    n_steps=250,
    n_samples=100,
    batch_size=512,
)

x_pred_tensor = torch.stack(x_pred, dim=1)

x_pred_mean = torch.mean(x_pred_tensor, dim=1)

B = x_pred_tensor.shape[0]
n_samples = x_pred_tensor.shape[1]
idx = torch.randint(0, n_samples, (B,), device=x_pred_tensor.device)
m_hat = x_pred_tensor[torch.arange(B, device=x_pred_tensor.device), idx]

y_test_scaled_np = y_test.detach().cpu().numpy()
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled_np)
x_pred_rescaled = scaler_y.inverse_transform(x_pred_mean.detach().cpu().numpy())

np.save("y_test_rescaled.npy", y_test_rescaled)
np.save("x_pred_rescaled.npy", x_pred_rescaled)

if y_test_rescaled.ndim == 2 and y_test_rescaled.shape[1] == 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test_rescaled[:, 0], x_pred_rescaled[:, 0], alpha=0.5)
    axes[0].set_title("Target 0")
    axes[1].scatter(y_test_rescaled[:, 1], x_pred_rescaled[:, 1], alpha=0.5)
    axes[1].set_title("Target 1")
else:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_rescaled.ravel(), x_pred_rescaled.ravel(), alpha=0.5)

y_test = y_test_rescaled
x_pred = x_pred_rescaled

if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)
if x_pred.ndim == 1:
    x_pred = x_pred.reshape(-1, 1)

if y_test.shape != x_pred.shape:
    raise ValueError(f"Shape mismatch: y_test {y_test.shape} vs x_pred {x_pred.shape}")

if y_test.shape[1] != 2:
    raise ValueError(f"Expected 2 targets for 2D histograms, got {y_test.shape[1]}")

# Native y is in MeV, so convert to GeV here (set to 1.0 if already in GeV).
unit_scale = 1e-3
y_true_mass = y_test[:, 0] * unit_scale
y_pred_mass = x_pred[:, 0] * unit_scale
y_true_ht = y_test[:, 1] * unit_scale
y_pred_ht = x_pred[:, 1] * unit_scale

bins_2d_mass = 250
bins_2d_ht = 250


mass_range = [[0, 6000], [0, 6000]]
ht_range = [[0, 4000], [0, 4000]]

error_mass_gev = (y_pred_mass - y_true_mass)
error_ht_gev = (y_pred_ht - y_true_ht)

err_counts_mass, err_bins_mass = np.histogram(error_mass_gev, bins=200, range=(-1e3, 1e3))
mode_error_mass = 0.5 * (err_bins_mass[:-1] + err_bins_mass[1:])[np.argmax(err_counts_mass)]
err_counts_ht, err_bins_ht = np.histogram(error_ht_gev, bins=200, range=(-1e3, 1e3))
mode_error_ht = 0.5 * (err_bins_ht[:-1] + err_bins_ht[1:])[np.argmax(err_counts_ht)]

counts_mass_2d, _, _ = np.histogram2d(
    y_true_mass, y_pred_mass, bins=bins_2d_mass, range=mass_range
)
counts_ht_2d, _, _ = np.histogram2d(
    y_true_ht, y_pred_ht, bins=bins_2d_ht, range=ht_range
)
max_count = max(counts_mass_2d.max(), counts_ht_2d.max())
norm_shared = mcolors.Normalize(vmin=0, vmax=max_count if max_count > 0 else 1)

def pearson_r(a, b):
    if a.size == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

mass_r = pearson_r(y_true_mass, y_pred_mass)
mass_rmse = rmse(y_true_mass, y_pred_mass)
mass_bias = mode_error_mass

ht_r = pearson_r(y_true_ht, y_pred_ht)
ht_rmse = rmse(y_true_ht, y_pred_ht)
ht_bias = mode_error_ht

box_kws = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black")

with plt.rc_context({"font.size": 20}):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    h_mass = axes[0].hist2d(
        y_true_mass,
        y_pred_mass,
        bins=bins_2d_mass,
        range=mass_range,
        cmap="hot",
        norm=norm_shared,
        cmin=1,
    )
    axes[0].plot([0, 6000], [0, 6000], color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    axes[0].set_xlabel("Expected Invariant Mass [GeV]")
    axes[0].set_ylabel("Predicted Invariant Mass [GeV]")
    textstr_mass = "\n".join([
        f"Pearson R: {mass_r:.4f}",
        f"RMSE: {mass_rmse:.2f} GeV",
        f"Bias: {mass_bias:.2f} GeV",
    ])
    axes[0].text(0.02, 0.98, textstr_mass, transform=axes[0].transAxes,
                    fontsize=20, va="top", bbox=box_kws, color="black")

    h_ht = axes[1].hist2d(
        y_true_ht,
        y_pred_ht,
        bins=bins_2d_ht,
        range=ht_range,
        cmap="hot",
        norm=norm_shared,
        cmin=1,
    )
    axes[1].plot([0, 4000], [0, 4000], color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Expected $H_T$ [GeV]")
    axes[1].set_ylabel("Predicted $H_T$ [GeV]")
    textstr_ht = "\n".join([
        f"Pearson R: {ht_r:.4f}",
        f"RMSE: {ht_rmse:.2f} GeV",
        f"Bias: {ht_bias:.2f} GeV",
    ])
    axes[1].text(0.02, 0.98, textstr_ht, transform=axes[1].transAxes,
                    fontsize=20, va="top", bbox=box_kws, color="black")

    cbar = fig.colorbar(h_mass[3], ax=axes, label="Counts")
    cbar.ax.tick_params(labelsize=18)

    handles_2d = axes[0].get_legend_handles_labels()
    if handles_2d[0]:
        fig.legend(handles_2d[0], handles_2d[1], loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=18)

    fig.savefig("Flow_Matching_mass_ht_2d_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def scaled_mass_to_gev(mass_scaled, ht_scaled_value, scaler, unit_scale_value):
    # Map a 1D mass grid in scaled space to GeV using the fixed HT context.
    arr = np.column_stack([mass_scaled, np.full_like(mass_scaled, ht_scaled_value)])
    mass_mev = scaler.inverse_transform(arr)[:, 0]
    return mass_mev * unit_scale_value

def gev_mass_to_scaled(mass_gev, ht_preproc_value, scaler_y, unit_scale_value):
    # Map a 1D mass grid in GeV to scaled space using a fixed HT in preprocessed units.
    mass_mev = mass_gev / unit_scale_value
    arr = np.column_stack([mass_mev, np.full_like(mass_mev, ht_preproc_value)])
    return scaler_y.transform(arr)[:, 0]

@torch.no_grad()
def plot_flow_panel_1d(model, embedder, X_test, pad_mask, y_test_scaled_np, y_true_mass, y_pred_mass, unit_scale_value,
                       n_noise=20000, nx=100, nt=100, device="cuda"):
    model.eval()
    embedder.eval()

    c = embedder(X_test[:1].to(device), pad_mask[:1].to(device)).float()
    # Keep HT fixed so the plot reflects only invariant mass across t.
    ht_scaled_mean = 0.0
    ht_preproc_mean = float(scaler_y.mean_[1])

    mass_grid_gev = np.linspace(0.0, 6e3, nx)
    mass_scaled_grid = gev_mass_to_scaled(mass_grid_gev, ht_preproc_mean, scaler_y, unit_scale_value)
    x_grid = torch.from_numpy(mass_scaled_grid).to(device=device, dtype=torch.float32)
    t_grid = torch.linspace(0.0, 1.0, nt, device=device, dtype=torch.float32)

    v_mass = []
    for t in t_grid:
        x_in = torch.stack([x_grid, torch.full_like(x_grid, ht_scaled_mean)], dim=1).float()
        t_in = torch.full((x_grid.shape[0], 1), float(t), device=device, dtype=torch.float32)
        c_in = c.repeat(x_grid.shape[0], 1).float()
        v_out = model(x_in, t_in, c_in)[:, 0]
        v_mass.append(v_out)
    v_mass = torch.stack(v_mass, dim=0).cpu().numpy()

    # Use a symmetric, robust normalization to avoid edge-row saturation.
    v_mass = np.nan_to_num(v_mass, nan=0.0, posinf=0.0, neginf=0.0)
    # Diagnostics for edge rows (mass boundaries).
    row_bottom = v_mass[:, 0]
    row_top = v_mass[:, -1]
    print(
        "[velocity diag] bottom row: min={:.4g} max={:.4g} p1={:.4g} p99={:.4g}".format(
            float(row_bottom.min()),
            float(row_bottom.max()),
            float(np.percentile(row_bottom, 1.0)),
            float(np.percentile(row_bottom, 99.0)),
        )
    )
    print(
        "[velocity diag] top row:    min={:.4g} max={:.4g} p1={:.4g} p99={:.4g}".format(
            float(row_top.min()),
            float(row_top.max()),
            float(np.percentile(row_top, 1.0)),
            float(np.percentile(row_top, 99.0)),
        )
    )
    vmax = np.percentile(np.abs(v_mass), 99.0)
    if vmax == 0.0:
        vmax = 1.0
    v_norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    x0 = torch.randn(n_noise, target_dim, device=device)
    x0_mass_gev = scaled_mass_to_gev(x0[:, 0].cpu().numpy(), ht_scaled_mean, scaler_y, unit_scale_value)

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1.2, 3.2, 1.2], figure=fig)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    ax_left.hist(x0_mass_gev, bins=100, range=(0, 6000), orientation="horizontal", color="tab:orange", alpha=0.7)
    ax_left.invert_xaxis()
    ax_left.set_title("Noise samples $x_0$")
    ax_left.set_xlabel("density")
    ax_left.set_ylabel("Invariant Mass [GeV]")

    im = ax_mid.imshow(
        v_mass.T,
        aspect="auto",
        origin="lower",
        extent=[0.0, 1.0, 0.0, 6000],
        cmap="coolwarm",
        norm=v_norm,
        interpolation="nearest",
    )
    ax_mid.set_title("Predicted velocity field $v(x,t)$")
    ax_mid.set_xlabel("$t$")

    ax_right.hist(y_true_mass, bins=100, range=(0, 6000), orientation="horizontal", color="tab:blue", alpha=0.6, label="True")
    ax_right.hist(y_pred_mass, bins=100, range=(0, 6000), orientation="horizontal", color="tab:red", alpha=0.6, label="Reco")
    ax_right.set_title("Truth vs Reco")
    ax_right.set_xlabel("density")
    ax_right.legend()

    cbar = fig.colorbar(
        im,
        ax=[ax_left, ax_mid, ax_right],
        orientation="horizontal",
        location="bottom",
        pad=0.08,
    )
    cbar.set_label("velocity")

    fig.savefig("Flow_Matching_velocity_field_panel.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

plot_flow_panel_1d(
    VelNet,
    Embedder,
    X_test,
    mask_test,
    y_test_scaled_np,
    y_true_mass,
    y_pred_mass,
    unit_scale,
    device=device,
)

bins = 100
x_min, x_max = 0, 6e3
x_min_ht, x_max_ht = 0, 4e3

y_true_mass_gev = y_true_mass
y_pred_mass_gev = y_pred_mass
y_true_ht_gev = y_true_ht
y_pred_ht_gev = y_pred_ht

invariant_mass_corr = pearson_r(y_true_mass_gev, y_pred_mass_gev)
invariant_mass_rmse = rmse(y_true_mass_gev, y_pred_mass_gev)
ht_corr = pearson_r(y_true_ht_gev, y_pred_ht_gev)
ht_rmse = rmse(y_true_ht_gev, y_pred_ht_gev)

with plt.rc_context({'font.size': 16}):
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[3, 1], hspace=0.05, wspace=0.25, figure=fig)
    ax_main_mass = fig.add_subplot(gs[0, 0])
    ax_main_ht = fig.add_subplot(gs[0, 1])
    ax_ratio_mass = fig.add_subplot(gs[1, 0], sharex=ax_main_mass)
    ax_ratio_ht = fig.add_subplot(gs[1, 1], sharex=ax_main_ht)

    main_kwargs = dict(bins=bins, histtype='step', linewidth=3)
    mass_true = ax_main_mass.hist(y_true_mass_gev, range=(x_min, x_max), color='blue', label='True', **main_kwargs)
    mass_pred = ax_main_mass.hist(y_pred_mass_gev, range=(x_min, x_max), color='red', label='Predicted', **main_kwargs)
    ht_true = ax_main_ht.hist(y_true_ht_gev, range=(x_min_ht, x_max_ht), color='blue', **main_kwargs)
    ht_pred = ax_main_ht.hist(y_pred_ht_gev, range=(x_min_ht, x_max_ht), color='red', **main_kwargs)

    # Add statistical errors (Poisson) for mass
    counts_true_mass, bin_edges_mass = np.histogram(y_true_mass_gev, bins=bins, range=(x_min, x_max))
    counts_pred_mass, _ = np.histogram(y_pred_mass_gev, bins=bins, range=(x_min, x_max))
    bin_centers_mass_plot = 0.5 * (bin_edges_mass[:-1] + bin_edges_mass[1:])
    errors_true_mass = np.sqrt(counts_true_mass)
    errors_pred_mass = np.sqrt(counts_pred_mass)
    ax_main_mass.errorbar(bin_centers_mass_plot, counts_true_mass, yerr=errors_true_mass, fmt='none', ecolor='blue', elinewidth=1, capsize=3, alpha=0.6)
    ax_main_mass.errorbar(bin_centers_mass_plot, counts_pred_mass, yerr=errors_pred_mass, fmt='none', ecolor='red', elinewidth=1, capsize=3, alpha=0.6)
    
    # Add statistical errors (Poisson) for HT
    counts_true_ht, bin_edges_ht = np.histogram(y_true_ht_gev, bins=bins, range=(x_min_ht, x_max_ht))
    counts_pred_ht, _ = np.histogram(y_pred_ht_gev, bins=bins, range=(x_min_ht, x_max_ht))
    bin_centers_ht_plot = 0.5 * (bin_edges_ht[:-1] + bin_edges_ht[1:])
    errors_true_ht = np.sqrt(counts_true_ht)
    errors_pred_ht = np.sqrt(counts_pred_ht)
    ax_main_ht.errorbar(bin_centers_ht_plot, counts_true_ht, yerr=errors_true_ht, fmt='none', ecolor='blue', elinewidth=1, capsize=3, alpha=0.6)
    ax_main_ht.errorbar(bin_centers_ht_plot, counts_pred_ht, yerr=errors_pred_ht, fmt='none', ecolor='red', elinewidth=1, capsize=3, alpha=0.6)

    ax_main_mass.set_ylabel('Frequency')

    textstr_mass = '\n'.join([
        f'Pearson R: {invariant_mass_corr:.4f}',
        f'RMSE: {invariant_mass_rmse:.2f} GeV'
    ])
    ax_main_mass.text(0.98, 0.98, textstr_mass, transform=ax_main_mass.transAxes,
                        fontsize=16, va='top', ha='right', bbox=box_kws, color='black')

    textstr_ht = '\n'.join([
        f'Pearson R: {ht_corr:.4f}',
        f'RMSE: {ht_rmse:.2f} GeV'
    ])
    ax_main_ht.text(0.98, 0.98, textstr_ht, transform=ax_main_ht.transAxes,
                    fontsize=16, va='top', ha='right', bbox=box_kws, color='black')

    ax_main_mass.grid(True, alpha=0.2)
    ax_main_ht.grid(True, alpha=0.2)

    # Ratio plot for mass
    ratio_mass = np.divide(counts_pred_mass, counts_true_mass, where=counts_true_mass != 0)
    ratio_mass[np.isnan(ratio_mass)] = 0.0
    ax_ratio_mass.plot(bin_centers_mass_plot, ratio_mass, color='black', linewidth=1.2)
    ax_ratio_mass.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax_ratio_mass.set_ylabel('Pred/True')
    ax_ratio_mass.set_xlabel('Invariant Mass [GeV]')
    ax_ratio_mass.set_ylim(0, 2)
    ax_ratio_mass.grid(True, alpha=0.2, axis='y')

    # Ratio plot for HT
    ratio_ht = np.divide(counts_pred_ht, counts_true_ht, where=counts_true_ht != 0)
    ratio_ht[np.isnan(ratio_ht)] = 0.0
    ax_ratio_ht.plot(bin_centers_ht_plot, ratio_ht, color='black', linewidth=1.2)
    ax_ratio_ht.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax_ratio_ht.set_ylabel('Pred/True')
    ax_ratio_ht.set_xlabel('$H_T$ [GeV]')
    ax_ratio_ht.set_ylim(0, 2)
    ax_ratio_ht.grid(True, alpha=0.2, axis='y')

    plt.setp(ax_main_mass.get_xticklabels(), visible=False)
    plt.setp(ax_main_ht.get_xticklabels(), visible=False)

    handles = [mass_true[2][0], mass_pred[2][0]]
    fig.legend(handles, ['True', 'Predicted'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=18)

    fig.savefig('Flow_Matching_mass_ht_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)