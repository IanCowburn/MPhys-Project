# Imports
from MPhys4TopRegressionTransformerModel import TransformerRegressor
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class TransformerScaling:
    def __init__(self, X, pad_mask_np, y,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets):
        self.X = X
        self.pad_mask_np = pad_mask_np
        self.y = y
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets

    def scale_y(self, y_train, y_valid, y_test):
        scaler_y = StandardScaler()
        y_train = torch.from_numpy(scaler_y.fit_transform(y_train.numpy().reshape(-1, 1)).reshape(-1)).float()
        y_valid = torch.from_numpy(scaler_y.transform(y_valid.numpy().reshape(-1, 1)).reshape(-1)).float()
        y_test = torch.from_numpy(scaler_y.transform(y_test.numpy().reshape(-1, 1)).reshape(-1)).float()
        return y_train, y_valid, y_test, scaler_y

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

    def scale_X(self, X_tensor, valid_tokens_tensor, scaler):
        # X_tensor: (E,T,F), valid_tokens_tensor: (E,T) bool
        X = X_tensor.numpy()
        valid_tokens = valid_tokens_tensor.numpy()
        E, T, F = X.shape
        X_flat = X.reshape(-1, F)
        valid_rows = valid_tokens.reshape(-1)
        X_flat_scaled = np.full_like(X_flat, -99.0, dtype=np.float32)
        X_flat_scaled[valid_rows] = scaler.transform(X_flat[valid_rows])
        return torch.from_numpy(X_flat_scaled.reshape(E, T, F)).float()

    def __call__(self):
        (X_train, X_valid, X_test,
         y_train, y_valid, y_test,
         mask_train, mask_valid, mask_test,
         valid_train, valid_valid, valid_test) = self.prepare_data(self.X, self.y, self.pad_mask_np)

        y_train, y_valid, y_test, scaler_y = self.scale_y(y_train, y_valid, y_test)
        
        X_train_np = X_train.numpy()
        valid_train_np = valid_train.numpy()

        X_train_flat = X_train_np.reshape(-1, X_train_np.shape[2])
        scalerX= StandardScaler()
        scalerX.fit(X_train_flat[valid_train_np.reshape(-1)])
        X_train_flat_scaled = np.full_like(X_train_flat, -99.0, dtype=np.float32)
        X_train_flat_scaled[valid_train_np.reshape(-1)] = scalerX.transform(X_train_flat[valid_train_np.reshape(-1)])
        X_train = torch.from_numpy(X_train_flat_scaled.reshape(X_train.shape)).float()
        
        X_valid = self.scale_X(X_valid, valid_valid, scalerX)
        X_test  = self.scale_X(X_test,  valid_test, scalerX)

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
                 X_train, X_valid, X_test,
                 mask_train, mask_valid, mask_test,
                 y_train, y_valid, y_test,
                 input_dim, embed_dim, n_heads, num_layers,
                 epochs, batch_size):
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets
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
            self.in_features_jets
        ).to(device)
        return model

    def training_loop(self, model, loss_fn, device, train_loader, valid_loader):
        early = EarlyStopping(patience=5, min_delta=1e-4)
        for epoch in range(1, self.epochs + 1):
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4 * (0.95 ** epoch))
            model.train()
            train_loss_sum = 0.0
            for xb, yb, mb in train_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb, padding_mask=mb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb, mb in valid_loader:
                    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                    pred = model(xb, padding_mask=mb)
                    loss = loss_fn(pred, yb)
                    val_loss_sum += loss.item() * xb.size(0)
            val_loss = val_loss_sum / len(valid_loader.dataset)
            print(f"Epoch {epoch}/{self.epochs}  Train {train_loss:.4f}  Val {val_loss:.4f}")
            early(val_loss)
            if early.early_stop:
                print("Early stopping")
                break
        return model

    def __call__(self):
        self.training_prints()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        train_loader, valid_loader, test_loader = self.build_loaders()
        model = self.init_model(device)
        loss_fn = nn.HuberLoss()
        model = self.training_loop(model, loss_fn, device, train_loader, valid_loader)
        return model, test_loader, device


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