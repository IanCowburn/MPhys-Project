# Imports
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

class TransformerEvaluation():
    """
    Evaluation Class for the Transformer Regressor model.
    """
    def __init__(self, model, test_loader, scaler_y, device, batch_size):
        """
        Initialise the evaluation class with model, test data loader, scaler, device, and batch size.
        
        Args:
            model (nn.Module): Trained transformer regression model.
            test_loader (DataLoader): DataLoader for the test dataset.
            scaler_y (sklearn.preprocessing.StandardScaler): Scaler for the target variables.
            device (torch.device): Device to run the evaluation on (CPU or GPU).
            batch_size (int): Batch size for evaluation.

        Returns:
            None
        """

        self.device = device
        self.model = model.to(device)
        self.test_loader = test_loader
        self.scaler_y = scaler_y
        self.batch_size = batch_size

    def evaluate(self):
        """
        Evaluate the model on the test dataset and return predictions and true values.
        
        Returns:
            y_pred_scaled (np.ndarray): Scaled predicted target values.
            y_true_scaled (np.ndarray): Scaled true target values.
            embedder_output (np.ndarray): Output from the embedder layer for analysis.
        """

        self.model.eval()

        y_pred_list = []
        y_true_list = []
        embedder_output = None

        with torch.no_grad():
            for xb, yb, mb in self.test_loader:  # X, y, mask
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)
                store = (len(y_pred_list) + 1) * self.batch_size >= len(self.test_loader.dataset)
                pred = self.model(xb, padding_mask=mb, store_embedder_output=store)
                if store and hasattr(self.model, 'stored_embedder_output'):
                    embedder_output = self.model.stored_embedder_output.detach().cpu().numpy()
                y_pred_list.append(pred.detach().cpu())
                y_true_list.append(yb.detach().cpu())

        y_pred_scaled = torch.cat(y_pred_list).numpy()
        y_true_scaled = torch.cat(y_true_list).numpy()
        return y_pred_scaled, y_true_scaled, embedder_output
    
    def compute_metrics(self, y_pred_scaled, y_true_scaled):
        """
        Compute evaluation metrics for the model predictions.
        
        Args:
            y_pred_scaled (np.ndarray): Scaled predicted target values.
            y_true_scaled (np.ndarray): Scaled true target values.

        Returns:
            y_true (np.ndarray): Inverse transformed true target values.
            y_pred (np.ndarray): Inverse transformed predicted target values.
            invariant_mass_corr (float): Correlation coefficient for invariant mass predictions.
            invariant_mass_rmse (float): RMSE for invariant mass predictions.
            invariant_mass_difference (np.ndarray): Differences between predicted and true invariant mass.
            ht_corr (float): Correlation coefficient for HT predictions.
            ht_rmse (float): RMSE for HT predictions.
            ht_difference (np.ndarray): Differences between predicted and true HT.
        """
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = y_pred.reshape(y_pred_scaled.shape)
        y_true = y_true.reshape(y_true_scaled.shape)

        # biased_difference = y_pred_biased - y_true
        # mean_difference = np.mean(biased_difference)
        # y_pred = y_pred_biased - mean_difference  # bias correction

        invariant_mass_difference = y_pred[:,0] - y_true[:,0]
        ht_difference = y_pred[:,1] - y_true[:,1]

        # mse_biased = mean_squared_error(y_true, y_pred_biased)
        # rmse_biased = np.sqrt(mse_biased)

        invariant_mass_mse = mean_squared_error(y_true[:,0], y_pred[:,0])
        invariant_mass_rmse = np.sqrt(invariant_mass_mse)

        ht_mse = mean_squared_error(y_true[:,1], y_pred[:,1])
        ht_rmse = np.sqrt(ht_mse)

        invariant_mass_corr = np.corrcoef(y_true[:,0], y_pred[:,0])[0, 1]
        print(f"Invariant Mass Correlation coefficient: {invariant_mass_corr:.4f}")

        ht_corr = np.corrcoef(y_true[:,1], y_pred[:,1])[0, 1]
        print(f"HT Correlation coefficient: {ht_corr:.4f}")

        # print(f"Biased RMSE: {rmse_biased:.2f} MeV")

        print(f"Unbiased RMSE: {invariant_mass_rmse:.2f} MeV")
        print(f"HT Unbiased RMSE: {ht_rmse:.2f} MeV")
        return y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference
    
    def __call__(self):
        """
        Perform evaluation and compute metrics.

        Returns:
            y_true (np.ndarray): Inverse transformed true target values.
            y_pred (np.ndarray): Inverse transformed predicted target values.
            invariant_mass_corr (float): Correlation coefficient for invariant mass predictions
            invariant_mass_rmse (float): RMSE for invariant mass predictions.
            invariant_mass_difference (np.ndarray): Differences between predicted and true invariant mass.
            ht_corr (float): Correlation coefficient for HT predictions.
            ht_rmse (float): RMSE for HT predictions.
            ht_difference (np.ndarray): Differences between predicted and true HT.
            embedder_output (np.ndarray): Stored embedder output from the model.
        """
        y_pred_scaled, y_true_scaled, embedder_output = self.evaluate()
        y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference = self.compute_metrics(y_pred_scaled, y_true_scaled)
        return y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, embedder_output
