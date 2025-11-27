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
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        # biased_difference = y_pred_biased - y_true
        # mean_difference = np.mean(biased_difference)
        # y_pred = y_pred_biased - mean_difference  # bias correction
        difference = y_pred - y_true  # recalculate difference
        # mse_biased = mean_squared_error(y_true, y_pred_biased)
        # rmse_biased = np.sqrt(mse_biased)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        print(f"Correlation coefficient: {corr:.4f}")

        # print(f"Biased RMSE: {rmse_biased:.2f} MeV")
        print(f"Unbiased RMSE: {rmse:.2f} MeV")

        return y_true, y_pred, corr, None, rmse, difference

    def __call__(self):
        y_pred_scaled, y_true_scaled, embedder_output = self.evaluate()
        y_true, y_pred, corr, rmse_biased, rmse, difference = self.compute_metrics(y_pred_scaled, y_true_scaled)
        return y_true, y_pred, corr, rmse_biased, rmse, difference, embedder_output