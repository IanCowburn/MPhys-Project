# Imports
from FourTopDualRegressorModel import TransformerRegressor
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class TransformerScaling:
    """
    Scaling Class for the Transformer Regressor model.
    """
    def __init__(self, X, pad_mask_np, y,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets):
        """
        Initialise the scaling class with data and parameters.

        Args:
            X (np.ndarray): Input feature array.
            pad_mask_np (np.ndarray): Padding mask array.
            y (np.ndarray): Target variable array.
            lepton_mask_size (int): Maximum number of leptons allowed per event.
            jet_mask_size (int): Maximum number of jets allowed per event.
            in_features_leptons (int): Number of input features for leptons.
            in_features_jets (int): Number of input features for jets.

        Returns:
            None
        """
        self.X = X
        self.pad_mask_np = pad_mask_np
        self.y = y
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.in_features_leptons = in_features_leptons
        self.in_features_jets = in_features_jets

    def scale_y(self, y_train, y_valid, y_test):
        """
        Scale the target variable y using StandardScaler.
        
        Args:
            y_train (torch.Tensor): Training target variable.
            y_valid (torch.Tensor): Validation target variable.
            y_test (torch.Tensor): Test target variable.
        
        Returns:
            y_train_scaled (torch.Tensor): Scaled training target variable.
            y_valid_scaled (torch.Tensor): Scaled validation target variable.
            y_test_scaled (torch.Tensor): Scaled test target variable.
            scaler_y (StandardScaler): Fitted scaler object.
        """
        scaler_y = StandardScaler()
        y_train_flat = y_train.reshape(-1, 1)
        y_valid_flat = y_valid.reshape(-1, 1)
        y_test_flat  = y_test.reshape(-1, 1)
        
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
        """
        Prepare the data by splitting into training, validation, and test sets.
        
        Args:
            X (np.ndarray): Input feature array.
            y_scaled (np.ndarray): Scaled target variable array.
            pad_mask_np (np.ndarray): Padding mask array.
        
        Returns:
            Tuple containing training, validation, and test splits for X, y, padding mask, and valid tokens.
        """
        valid_tokens = ~pad_mask_np  # numpy (E,T)

        # Make everything tensors before split so train_test_split returns tensors consistently
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y_scaled).float()
        pad_t = torch.from_numpy(pad_mask_np).bool()
        valid_t = torch.from_numpy(valid_tokens).bool()

        X_train, X_test, y_train, y_test, mask_train, mask_test, valid_train, valid_test = train_test_split(
            X_t, y_t, pad_t, valid_t, test_size=0.2, random_state=42)

        X_valid, X_test, y_valid, y_test, mask_valid, mask_test, valid_valid, valid_test = train_test_split(
            X_test, y_test, mask_test, valid_test, test_size=0.5, random_state=42)

        return (X_train, X_valid, X_test,
                y_train, y_valid, y_test,
                mask_train, mask_valid, mask_test,
                valid_train, valid_valid, valid_test)
    
    def scale_X(self, X_tensor, valid_tokens_tensor, scaler):
        """
        Scale the input features X using the provided scaler, only for valid tokens.

        Args:
            X_tensor (torch.Tensor): Input feature tensor.
            valid_tokens_tensor (torch.Tensor): Valid tokens mask tensor.
            scaler (StandardScaler): Fitted scaler object.
        
        Returns:
            torch.Tensor: Scaled input feature tensor.
        """
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
        """
        Execute the scaling and data preparation process.

        Returns:
            Tuple containing training, validation, and test splits for X, y, padding mask, and the fitted scaler for y.
        """
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
    """
    Training Class for the Transformer Regressor model.
    """
    def __init__(self,
                 lepton_mask_size, jet_mask_size,
                 in_features_leptons, in_features_jets,
                 X_train, X_valid, X_test,
                 mask_train, mask_valid, mask_test,
                 y_train, y_valid, y_test,
                 input_dim, embed_dim, n_heads, num_layers,
                 epochs, batch_size):
        """
        Initialize the TransformerTraining class with model and training parameters.
        
        Args:
            lepton_mask_size (int): Maximum number of leptons allowed per event.
            jet_mask_size (int): Maximum number of jets allowed per event.
            in_features_leptons (int): Number of input features for leptons.
            in_features_jets (int): Number of input features for jets.
            X_train (torch.Tensor): Training input features.
            X_valid (torch.Tensor): Validation input features.
            X_test (torch.Tensor): Test input features.
            mask_train (torch.Tensor): Training padding mask.
            mask_valid (torch.Tensor): Validation padding mask.
            mask_test (torch.Tensor): Test padding mask.
            y_train (torch.Tensor): Training target variable.
            y_valid (torch.Tensor): Validation target variable.
            y_test (torch.Tensor): Test target variable.
            input_dim (int): Dimension of the input features.
            embed_dim (int): Dimension of the embedding space.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.

        Returns:
            None
        """
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
        """
        Print the shapes of the training, validation, and test datasets.
        """
        print("X_train:", self.X_train.shape)
        print("X_valid:", self.X_valid.shape)
        print("X_test :", self.X_test.shape)

    def build_loaders(self):
        """
        Build DataLoader objects for training, validation, and test datasets.
        """
        train_ds = TensorDataset(self.X_train, self.y_train, self.mask_train)
        valid_ds = TensorDataset(self.X_valid, self.y_valid, self.mask_valid)
        test_ds  = TensorDataset(self.X_test,  self.y_test,  self.mask_test)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    
    def init_model(self, device):
        """
        Initialise the Transformer Regressor model.

        Args:
            device (torch.device): Device to run the model on (CPU or GPU).

        Returns:
            model (nn.Module): Initialized Transformer Regressor model.
        """
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
        """
        Training loop for the Transformer Regressor model.

        Args:
            model (nn.Module): Transformer Regressor model.
            loss_fn (nn.Module): Loss function.
            device (torch.device): Device to run the training on (CPU or GPU).
            train_loader (DataLoader): DataLoader for the training dataset.
            valid_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            model (nn.Module): Trained Transformer Regressor model.
        """

        early = EarlyStopping(patience=5, min_delta=1e-4)
        for epoch in range(1, self.epochs + 1):
            optimizer = torch.optim.Adam(model.parameters(), lr=4e-4 * (0.95 ** epoch))
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
        """
        Execute the training process.

        Returns:
            model (nn.Module): Trained Transformer Regressor model.
            test_loader (DataLoader): DataLoader for the test dataset.
            device (torch.device): Device used for training (CPU or GPU).
        """
        self.training_prints()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        train_loader, valid_loader, test_loader = self.build_loaders()
        model = self.init_model(device)
        loss_fn = nn.HuberLoss()
        model = self.training_loop(model, loss_fn, device, train_loader, valid_loader)
        return model, test_loader, device

class EarlyStopping:
    """
    Early Stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience=10, min_delta=0.0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        
        Returns:
            None
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to update early stopping status based on validation loss.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            None
        """
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
