# Imports
from TTBarDualRegressorModel import TransformerRegressor
import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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

class TransformerTraining:
    def __init__(self,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets,
                 in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu,
                 X_train, X_valid, X_test,
                 mask_train, mask_valid, mask_test,
                 y_train, y_valid, y_test,
                 input_dim, embed_dim, n_heads, num_layers,
                 epochs, batch_size, scaler_y, x_scalers):
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_ellipse_nu = in_features_ellipse_nu
        self.in_features_ellipse_anti_nu = in_features_ellipse_anti_nu
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.mask_train = mask_train
        self.mask_valid = mask_valid
        self.mask_test = mask_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_y = scaler_y
        self.x_scalers = x_scalers

    def training_prints(self):
        print("X_train:", self.X_train.shape)
        print("X_valid:", self.X_valid.shape)
        print("X_test :", self.X_test.shape)

    def build_loaders(self):
        # Extract MET px/py (unscaled) for physics loss
        # MET token is at index lepton_mask_size + jet_mask_size
        # Features: [met, phi, 0, 0, 0, 0] — we need px = met*cos(phi), py = met*sin(phi)
        # But features are already scaled, so we store raw MET px/py from X before scaling.
        # Instead, we pass MET info through the dataloader and compute in training loop.
        train_ds = TensorDataset(self.X_train, self.y_train, self.mask_train)
        valid_ds = TensorDataset(self.X_valid, self.y_valid, self.mask_valid)
        test_ds  = TensorDataset(self.X_test,  self.y_test,  self.mask_test)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    
    def init_model(self, device):
        model = TransformerRegressor(
            self.embed_dim,
            self.n_heads,
            self.num_layers,
            self.lepton_mask_size,
            self.jet_mask_size,
            self.in_features_leptons,
            self.in_features_jets,
            self.in_features_met,
            self.in_features_ellipse_nu,
            self.in_features_ellipse_anti_nu,
        ).to(device)
        return model
    
    def distribution_considering_loss(self, pred, target, bins, hist_min, hist_max, sigma=0.20, eps=1e-8):

        pred = pred.reshape(-1)
        target = target.reshape(-1)

        centers = torch.linspace(hist_min, hist_max, bins, device=pred.device, dtype=pred.dtype)

        pred_kernel = torch.exp(-0.5 * ((pred.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)
        target_kernel = torch.exp(-0.5 * ((target.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)

        pred_hist = pred_kernel.mean(dim=0) + eps
        target_hist = target_kernel.mean(dim=0) + eps

        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()

        kl_div = torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)))
        return kl_div
    
    def training_loop(self, model, device, train_loader, valid_loader):
        early = EarlyStopping(patience=20, min_delta=1e-4)
        loss_fn = nn.MSELoss() 
        train_loss_history = []
        val_loss_history = []
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2.5e-4,
            weight_decay=2e-4,  # scaled down
        )

        # Verify target range before training
        print(f"Target range: [{self.y_train.min():.2f}, {self.y_train.max():.2f}]")

        # KL settings: keep KL weak at start so regression can lock onto the target first.
        kl_weight_max = 0.3
        kl_ramp_epochs = 15

        # Build histogram range from train targets to avoid empty-bin instability.
        hist_min = torch.quantile(self.y_train, 0.001).item() - 0.25
        hist_max = torch.quantile(self.y_train, 0.999).item() + 0.25
        hist_bins = 64
        
        # Warmup + cosine annealing
        warmup_epochs = 10
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        for epoch in range(1, self.epochs + 1):
            current_kl_weight = kl_weight_max * min(1.0, epoch / kl_ramp_epochs)
            model.train()
            train_total_sum = 0.0
            train_mse_sum = 0.0
            train_kl_sum = 0.0
            for xb, yb, mb in train_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

                pred = model(xb, padding_mask=mb)
                mse_loss = loss_fn(pred, yb)
                kl_loss = self.distribution_considering_loss(
                    pred, yb, bins=hist_bins, hist_min=hist_min, hist_max=hist_max
                )

                # Combine losses
                total_loss = mse_loss + current_kl_weight * kl_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_total_sum += total_loss.item() * xb.size(0)
                train_mse_sum += mse_loss.item() * xb.size(0)
                train_kl_sum += kl_loss.item() * xb.size(0)

            train_total = train_total_sum / len(train_loader.dataset)
            train_mse = train_mse_sum / len(train_loader.dataset)
            train_kl = train_kl_sum / len(train_loader.dataset)

            model.eval()
            val_total_sum = 0.0
            val_mse_sum = 0.0
            val_kl_sum = 0.0
            with torch.no_grad():
                for xb, yb, mb in valid_loader:
                    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                    pred = model(xb, padding_mask=mb)
                    mse_loss = loss_fn(pred, yb)
                    kl_loss = self.distribution_considering_loss(
                        pred, yb, bins=hist_bins, hist_min=hist_min, hist_max=hist_max
                    )
                    total_loss = mse_loss + current_kl_weight * kl_loss

                    val_total_sum += total_loss.item() * xb.size(0)
                    val_mse_sum += mse_loss.item() * xb.size(0)
                    val_kl_sum += kl_loss.item() * xb.size(0)

            val_total = val_total_sum / len(valid_loader.dataset)
            val_mse = val_mse_sum / len(valid_loader.dataset)
            val_kl = val_kl_sum / len(valid_loader.dataset)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}/{self.epochs}  "
                f"TrainTot {train_total:.4f} (MSE: {train_mse:.4f}, KL: {train_kl:.4f})  "
                f"ValTot {val_total:.4f} (MSE: {val_mse:.4f}, KL: {val_kl:.4f})  "
                f"KLw {current_kl_weight:.3f}  "
                f"LR {current_lr:.2e}"
            )
            train_loss_history.append(train_total)
            val_loss_history.append(val_total)
            early(val_mse)
            if early.early_stop:
                print("Early stopping")
                break
        return model, train_loss_history, val_loss_history
    def __call__(self):
        self.training_prints()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        train_loader, valid_loader, test_loader = self.build_loaders()
        model = self.init_model(device)
        model, train_loss_history, val_loss_history = self.training_loop(model, device, train_loader, valid_loader)
        return model, test_loader, device, train_loss_history, val_loss_history

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
        self.early_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
