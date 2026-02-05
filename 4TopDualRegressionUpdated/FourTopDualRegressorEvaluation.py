
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
class TransformerEvaluation():
    def __init__(self, model, test_loader, scaler_y, device, batch_size):
        # Ensure model is on the correct device
        self.device = device
        self.model = model.to(device)
        self.test_loader = test_loader
        self.scaler_y = scaler_y
        self.batch_size = batch_size
    def evaluate(self):
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
        # Inverse transform to get physical invariant mass values
        y_pred_biased = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_true_scaled)

        #Unlog transform to get back to original scale
        y_pred_biased = np.exp(y_pred_biased)
        y_true = np.exp(y_true)

        y_pred = y_pred_biased # - mean_difference  # bias correction
        
        invariant_mass_difference = y_pred[:,0] - y_true[:,0]
        ht_difference = y_pred[:,1] - y_true[:,1]

        # Calculate mode errors
        err_counts_mass, err_bins_mass = np.histogram(invariant_mass_difference, bins=200, range=(-1e3, 1e3))
        mode_error_mass = 0.5 * (err_bins_mass[:-1] + err_bins_mass[1:])[np.argmax(err_counts_mass)]
        err_counts_ht, err_bins_ht = np.histogram(ht_difference, bins=200, range=(-1e3, 1e3))
        mode_error_ht = 0.5 * (err_bins_ht[:-1] + err_bins_ht[1:])[np.argmax(err_counts_ht)]

        invariant_mass_biased = y_pred[:,0] - mode_error_mass
        ht_biased = y_pred[:,1] - mode_error_ht

        invariant_mass_rmse_biased = np.sqrt(mean_squared_error(y_true[:,0], invariant_mass_biased))
        ht_rmse_biased = np.sqrt(mean_squared_error(y_true[:,1], ht_biased))


        invariant_mass_mse = mean_squared_error(y_true[:,0], y_pred[:,0])
        invariant_mass_rmse = np.sqrt(invariant_mass_mse)
        ht_mse = mean_squared_error(y_true[:,1], y_pred[:,1])
        ht_rmse = np.sqrt(ht_mse)
        invariant_mass_corr = np.corrcoef(y_true[:,0], y_pred[:,0])[0, 1]
        print(f"Invariant Mass Correlation coefficient: {invariant_mass_corr:.4f}")
        ht_corr = np.corrcoef(y_true[:,1], y_pred[:,1])[0, 1]
        print(f"HT Correlation coefficient: {ht_corr:.4f}")
        print(f"Invariant Mass RMSE after unbiasing: {invariant_mass_rmse_biased:.2f} MeV")
        print(f"HT RMSE after unbiasing: {ht_rmse_biased:.2f} MeV")
        print(f"Invariant Mass RMSE: {invariant_mass_rmse:.2f} MeV")
        print(f"HT RMSE: {ht_rmse:.2f} MeV")

        

        

        return y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased
    def __call__(self):
        y_pred_scaled, y_true_scaled, embedder_output = self.evaluate()
        y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased = self.compute_metrics(y_pred_scaled, y_true_scaled)
        return y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased, embedder_output