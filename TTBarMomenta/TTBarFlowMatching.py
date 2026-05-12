import torch as torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TTBarDualRegressorModel import TransformerEncoder, LeptonEmbedder, JetEmbedder, METEmbedder, EllipseNuEmbedder, EllipseAntiNuEmbedder
from TTBarDualRegressorPlotting import TransformerPlotting

X = np.load("ttbar_dual_cache_X.npy")
y = np.load("ttbar_dual_cache_y.npy")
pad_mask_np = np.load("ttbar_dual_cache_pad_mask.npy")

class TransformerScaling:
    def __init__(self, X, pad_mask_np, y,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu):
        self.X = X
        self.pad_mask_np = pad_mask_np
        self.y = y
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_ellipse_nu = in_features_ellipse_nu
        self.in_features_ellipse_anti_nu = in_features_ellipse_anti_nu
    def scale_y(self, y_train, y_valid, y_test):
        # Signed targets (px, py, pz) should not be log-transformed.
        scaler_y = StandardScaler()
        y_train_np = y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else np.asarray(y_train)
        y_valid_np = y_valid.detach().cpu().numpy() if torch.is_tensor(y_valid) else np.asarray(y_valid)
        y_test_np = y_test.detach().cpu().numpy() if torch.is_tensor(y_test) else np.asarray(y_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train_np)
        y_valid_scaled = scaler_y.transform(y_valid_np)
        y_test_scaled = scaler_y.transform(y_test_np)

        y_train_scaled = torch.from_numpy(y_train_scaled).float()
        y_valid_scaled = torch.from_numpy(y_valid_scaled).float()
        y_test_scaled = torch.from_numpy(y_test_scaled).float()
        
        return y_train_scaled, y_valid_scaled, y_test_scaled, scaler_y
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

        # Leptons and jets: scale kinematics (eta, phi, pt, e), keep feature 4 unchanged
        for token_slice in (lepton_slice, jet_slice):
            group = X[:, token_slice, :4]
            valid_rows = (~pad_mask[:, token_slice]).reshape(-1)
            flat_group = group.reshape(-1, 4)
            if np.any(valid_rows):
                flat_group[valid_rows] = scalers["lepjet"].transform(flat_group[valid_rows])
            X[:, token_slice, :4] = flat_group.reshape(group.shape)

        # MET token: scale first 4 channels
        X[:, met_idx, :4] = scalers["met"].transform(X[:, met_idx, :4])

        if self.in_features_ellipse_nu > 0:
            # Ellipse nu token: scale first 4 channels
            ellipse_nu_idx = met_idx + 1
            X[:, ellipse_nu_idx, :4] = scalers["ellipse_nu"].transform(X[:, ellipse_nu_idx, :4])

            # Ellipse anti-nu token: scale first 4 channels
            ellipse_anti_nu_idx = ellipse_nu_idx + 1
            X[:, ellipse_anti_nu_idx, :4] = scalers["ellipse_anti_nu"].transform(X[:, ellipse_anti_nu_idx, :4])

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
        }
        if self.in_features_ellipse_nu > 0:
            ellipse_nu_idx = met_idx + 1
            ellipse_anti_nu_idx = ellipse_nu_idx + 1
            scalers["ellipse_nu"] = StandardScaler().fit(X_train_np[:, ellipse_nu_idx, :4])
            scalers["ellipse_anti_nu"] = StandardScaler().fit(X_train_np[:, ellipse_anti_nu_idx, :4])

        X_train = self.scale_X(X_train, mask_train, scalers)
        X_valid = self.scale_X(X_valid, mask_valid, scalers)
        X_test = self.scale_X(X_test, mask_test, scalers)
        print(f"Training set:   {X_train.shape[0]} events")
        print(f"Validation set: {X_valid.shape[0]} events")
        print(f"Test set:       {X_test.shape[0]} events")
        return (X_train, X_valid, X_test,
                y_train, y_valid, y_test,
                mask_train, mask_valid, mask_test,
                scaler_y, scalers)

lepton_mask_size = 2
jet_mask_size = 8
in_features_jets = 5
in_features_leptons = 6
in_features_met = 2
in_features_ellipse_nu = 4
in_features_ellipse_anti_nu = 4
embed_dim = 256
n_heads = 8
num_layers = 6
epochs = 50
batch_size = 1024

scaler = TransformerScaling(X, pad_mask_np, y,
                            lepton_mask_size, jet_mask_size,
                            in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu)
(X_train, X_valid, X_test,
 y_train, y_valid, y_test,
 mask_train, mask_valid, mask_test,
 scaler_y, x_scalers) = scaler()

class TransformerRegressor(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu, output_dim=6):
        super().__init__()
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_ellipse_nu = in_features_ellipse_nu
        self.in_features_ellipse_anti_nu = in_features_ellipse_anti_nu
        self.has_ellipse = in_features_ellipse_nu > 0 and in_features_ellipse_anti_nu > 0
        self.lepton_embedder = LeptonEmbedder(in_features_leptons, embed_dim)
        self.jet_embedder = JetEmbedder(in_features_jets, embed_dim)
        self.met_embedder = METEmbedder(in_features_met, embed_dim)
        if self.has_ellipse:
            self.ellipse_nu_embedder = EllipseNuEmbedder(in_features_ellipse_nu, embed_dim)
            self.ellipse_anti_nu_embedder = EllipseAntiNuEmbedder(in_features_ellipse_anti_nu, embed_dim)
        n_token_types = 5 if self.has_ellipse else 3
        self.type_embedding = nn.Embedding(n_token_types, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embed_dim, n_heads, num_layers, dropout=0.15)
        # Attention-weighted pooling
        self.attn_pool = nn.Linear(embed_dim, 1)
        # Separate heads for nu and antinu with deeper residual MLP
        h = embed_dim // 2
        self.antinu_proj = nn.Linear(embed_dim, h)
        self.antinu_block = nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
        )
        self.antinu_out = nn.Linear(h, 3)  # antinu_px, antinu_py, antinu_pz
        self.nu_proj = nn.Linear(embed_dim, h)
        self.nu_block = nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.25),
        )
        self.nu_out = nn.Linear(h, 3)  # nu_px, nu_py, nu_pz
        self.last_embedder_output = None
    def forward(self, x, padding_mask=None, store_embedder_output=False):
        if padding_mask is None:
            padding_mask = (x == -99.0).all(dim=-1)
        
        leptons = x[:, :self.lepton_mask_size, :self.in_features_leptons]
        jets = x[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size, :self.in_features_jets]
        met = x[:, self.lepton_mask_size + self.jet_mask_size:self.lepton_mask_size + self.jet_mask_size + 1, :self.in_features_met]
        lepton_mask = padding_mask[:, :self.lepton_mask_size]
        jet_mask = padding_mask[:, self.lepton_mask_size:self.lepton_mask_size + self.jet_mask_size]
        lepton_embedded = self.lepton_embedder(leptons, lepton_mask)
        jet_embedded = self.jet_embedder(jets, jet_mask)
        met_embedded = self.met_embedder(met)
        if self.has_ellipse:
            ellipse_nu = x[:, self.lepton_mask_size + self.jet_mask_size + 1:self.lepton_mask_size + self.jet_mask_size + 2, :self.in_features_ellipse_nu]
            ellipse_anti_nu = x[:, self.lepton_mask_size + self.jet_mask_size + 2:self.lepton_mask_size + self.jet_mask_size + 3, :self.in_features_ellipse_anti_nu]
            ellipse_nu_embedded = self.ellipse_nu_embedder(ellipse_nu)
            ellipse_anti_nu_embedded = self.ellipse_anti_nu_embedder(ellipse_anti_nu)
            embedded = torch.cat((lepton_embedded, jet_embedded, met_embedded, ellipse_nu_embedded, ellipse_anti_nu_embedded), dim=1)
        else:
            embedded = torch.cat((lepton_embedded, jet_embedded, met_embedded), dim=1)
        batch_size = x.size(0)
        if self.has_ellipse:
            type_ids = torch.cat([
                torch.zeros((batch_size, self.lepton_mask_size), dtype=torch.long, device=x.device),
                torch.ones((batch_size, self.jet_mask_size), dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 2, dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 3, dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 4, dtype=torch.long, device=x.device),
            ], dim=1)
        else:
            type_ids = torch.cat([
                torch.zeros((batch_size, self.lepton_mask_size), dtype=torch.long, device=x.device),
                torch.ones((batch_size, self.jet_mask_size), dtype=torch.long, device=x.device),
                torch.full((batch_size, 1), 2, dtype=torch.long, device=x.device),
            ], dim=1)
        type_embedded = self.type_embedding(type_ids)
        embedded = self.embed_dropout(embedded + type_embedded)

        if store_embedder_output:
            self.last_embedder_output = embedded.detach()

        x = self.transformer(embedded, mask=padding_mask)
        # Attention-weighted pooling
        attn_weights = self.attn_pool(x).squeeze(-1)           # (batch, tokens)
        attn_weights = attn_weights.masked_fill(padding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)      # (batch, tokens)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)        # (batch, embed_dim)
        # Separate residual heads
        antinu_h = self.antinu_proj(x)
        antinu_h = antinu_h + self.antinu_block(antinu_h)
        antinu_pred = self.antinu_out(antinu_h)
        nu_h = self.nu_proj(x)
        nu_h = nu_h + self.nu_block(nu_h)
        nu_pred = self.nu_out(nu_h)
        out = torch.cat([antinu_pred, nu_pred], dim=1)  # (batch, 6)
        return out
    
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

device = torch.device("cuda")
N = X_train.shape[0]
batch_size = 256
num_epochs = 50
target_dim = y_train.shape[1]
Embedder = TransformerRegressor(embed_dim, n_heads, num_layers, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu, output_dim=6).to(device)

# Infer context dimensionality directly from the embedder output to avoid shape mismatches.
with torch.no_grad():
    context_dim = int(
        Embedder(
            X_train[:1].to(device),
            mask_train[:1].to(device),
        ).shape[1]
    )
VelNet = ConditionalVelocityNet(Ninputs=target_dim, Ncontext=context_dim, hidden=128).to(device)

optimizer = torch.optim.AdamW(list(VelNet.parameters()) + list(Embedder.parameters()), lr=2.5e-4, weight_decay=1e-4)
        # Warmup + cosine annealing
warmup_epochs = 10
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs - warmup_epochs
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
)

patience = 25
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
    scheduler.step()
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

y_true = y_test_rescaled
y_pred = x_pred_rescaled

if y_true.ndim == 1:
    y_true = y_true.reshape(-1, 1)
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

if y_true.shape != y_pred.shape:
    raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

if y_true.shape[1] != 6:
    raise ValueError(f"Expected 6 targets to match TTBarDualRegressorPlotting, got {y_true.shape[1]}")


def pearson_r(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 2:
        return 0.0
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def rmse(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) == 0:
        return 0.0
    return float(np.sqrt(np.mean((a[finite] - b[finite]) ** 2)))


differences = y_pred - y_true
component_names = [
    "antinu_px",
    "antinu_py",
    "antinu_pz",
    "nu_px",
    "nu_py",
    "nu_pz",
]

metrics = {}
for idx, name in enumerate(component_names):
    err = differences[:, idx]
    metrics[name] = {
        "corr": pearson_r(y_true[:, idx], y_pred[:, idx]),
        "rmse": rmse(y_true[:, idx], y_pred[:, idx]),
        "mean_error": float(np.nanmean(err)),
        "std_error": float(np.nanstd(err)),
    }

plotter = TransformerPlotting(
    y_true=y_true,
    y_pred=y_pred,
    metrics=metrics,
    differences=differences,
    train_loss_history=losses,
    val_loss_history=val_losses,
    suffix="_flow_matching",
)
plotter.plot_all()