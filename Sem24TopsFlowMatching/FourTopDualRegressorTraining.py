# Imports
from FourTopDualRegressorModel import TransformerRegressor
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
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
        # Try log transformation instead of just StandardScaler
        
        scaler_y = StandardScaler()
        y_train_flat = np.log(y_train).reshape(-1, 1)
        y_valid_flat = np.log(y_valid).reshape(-1, 1)
        y_test_flat  = np.log(y_test).reshape(-1, 1)
        
        y_train_scaled = scaler_y.fit_transform(y_train_flat).ravel()
        y_valid_scaled = scaler_y.transform(y_valid_flat).ravel()
        y_test_scaled = scaler_y.transform(y_test_flat).ravel()
        y_train_scaled = y_train_scaled.reshape(y_train.shape)
        y_valid_scaled = y_valid_scaled.reshape(y_valid.shape)
        y_test_scaled = y_test_scaled.reshape(y_test.shape)
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
        numbers_idx = met_idx + 1
        ht_idx = numbers_idx + 1

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

        # Numbers token: scale all 5 channels
        X[:, numbers_idx, :5] = scalers["numbers"].transform(X[:, numbers_idx, :5])

        # HT token: scale first 3 channels
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
        X_test = self.scale_X(X_test, mask_test, scalers)
        print(f"Training set:   {X_train.shape[0]} events")
        print(f"Validation set: {X_valid.shape[0]} events")
        print(f"Test set:       {X_test.shape[0]} events")
        return (X_train, X_valid, X_test,
                y_train, y_valid, y_test,
                mask_train, mask_valid, mask_test,
                scaler_y)

class TransformerTraining:
    def __init__(self,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets,
                 in_features_met, in_features_numbers, in_features_ht,
                 X_train, X_valid, X_test,
                 mask_train, mask_valid, mask_test,
                 y_train, y_valid, y_test,
                 input_dim, embed_dim, n_heads, num_layers,
                 epochs, batch_size):
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
        self.in_features_met = in_features_met
        self.in_features_numbers = in_features_numbers
        self.in_features_ht = in_features_ht
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

    def training_prints(self):
        print("X_train:", self.X_train.shape)
        print("X_valid:", self.X_valid.shape)
        print("X_test :", self.X_test.shape)

    def build_loaders(self):
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
            self.in_features_numbers,
            self.in_features_ht
        ).to(device)
        return model
    
    def distribution_considering_loss(self, pred, target, bins, hist_min, hist_max, sigma=0.20, eps=1e-8):
        # Handle both single-target [B] and multi-target [B, D] regression.
        if pred.ndim == 1:
            pred = pred.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(1)

        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: pred {pred.shape}, target {target.shape}")

        target_dim = pred.shape[1]
        if isinstance(hist_min, (float, int)):
            hist_min = pred.new_tensor([hist_min] * target_dim)
        if isinstance(hist_max, (float, int)):
            hist_max = pred.new_tensor([hist_max] * target_dim)

        hist_min = hist_min.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        hist_max = hist_max.to(device=pred.device, dtype=pred.dtype).reshape(-1)

        if hist_min.numel() != target_dim or hist_max.numel() != target_dim:
            raise ValueError(
                f"Histogram bounds must have {target_dim} values, got min={hist_min.numel()}, max={hist_max.numel()}"
            )

        kl_sum = 0.0
        for dim_idx in range(target_dim):
            pred_dim = pred[:, dim_idx]
            target_dim_values = target[:, dim_idx]

            centers = torch.linspace(
                hist_min[dim_idx], hist_max[dim_idx], bins,
                device=pred.device, dtype=pred.dtype
            )

            pred_kernel = torch.exp(-0.5 * ((pred_dim.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)
            target_kernel = torch.exp(-0.5 * ((target_dim_values.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)

            pred_hist = pred_kernel.mean(dim=0) + eps
            target_hist = target_kernel.mean(dim=0) + eps

            pred_hist = pred_hist / pred_hist.sum()
            target_hist = target_hist / target_hist.sum()

            kl_sum = kl_sum + torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)))

        return kl_sum / target_dim
    
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
        if self.y_train.ndim == 1:
            print(f"Target range: [{self.y_train.min():.2f}, {self.y_train.max():.2f}]")
        else:
            y_min = torch.min(self.y_train, dim=0).values
            y_max = torch.max(self.y_train, dim=0).values
            print(f"Target ranges per output: min={y_min.tolist()}, max={y_max.tolist()}")

        # KL settings: keep KL weak at start so regression can lock onto the target first.
        kl_weight_max = 0
        kl_ramp_epochs = 15

        # Build histogram range from train targets to avoid empty-bin instability.
        if self.y_train.ndim == 1:
            hist_min = torch.quantile(self.y_train, 0.001).item() - 0.25
            hist_max = torch.quantile(self.y_train, 0.999).item() + 0.25
        else:
            hist_min = torch.quantile(self.y_train, 0.001, dim=0) - 0.25
            hist_max = torch.quantile(self.y_train, 0.999, dim=0) + 0.25
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
                if pred.shape != yb.shape:
                    raise ValueError(f"Model output shape {pred.shape} does not match target shape {yb.shape}")
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
                    if pred.shape != yb.shape:
                        raise ValueError(f"Model output shape {pred.shape} does not match target shape {yb.shape}")
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