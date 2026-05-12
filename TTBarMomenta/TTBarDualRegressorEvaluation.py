import torch
import numpy as np
from sklearn.metrics import mean_squared_error


class TransformerEvaluation:
    def __init__(self, model, test_loader, scaler_y, device, batch_size):
        self.device = device
        self.model = model.to(device)
        self.test_loader = test_loader
        self.scaler_y = scaler_y
        self.batch_size = batch_size
        self.component_names = [
            "antinu_px",
            "antinu_py",
            "antinu_pz",
            "nu_px",
            "nu_py",
            "nu_pz",
        ]

    def evaluate(self):
        self.model.eval()
        y_pred_list = []
        y_true_list = []
        embedder_output = None

        with torch.no_grad():
            for xb, yb, mb in self.test_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)

                store = (len(y_pred_list) + 1) * self.batch_size >= len(self.test_loader.dataset)
                pred = self.model(xb, padding_mask=mb, store_embedder_output=store)

                if store and hasattr(self.model, "last_embedder_output"):
                    embedder_output = self.model.last_embedder_output.detach().cpu().numpy()

                y_pred_list.append(pred.detach().cpu())
                y_true_list.append(yb.detach().cpu())

        y_pred_scaled = torch.cat(y_pred_list).numpy()
        y_true_scaled = torch.cat(y_true_list).numpy()
        return y_pred_scaled, y_true_scaled, embedder_output

    def compute_metrics(self, y_pred_scaled, y_true_scaled):
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_true_scaled)

        differences = y_pred - y_true
        metrics = {}

        for index, name in enumerate(self.component_names):
            rmse = np.sqrt(mean_squared_error(y_true[:, index], y_pred[:, index]))
            true_std = np.std(y_true[:, index])
            pred_std = np.std(y_pred[:, index])
            if true_std < 1e-12 or pred_std < 1e-12:
                corr = 0.0
            else:
                corr = np.corrcoef(y_true[:, index], y_pred[:, index])[0, 1]
                if np.isnan(corr):
                    corr = 0.0

            metrics[name] = {
                "rmse": float(rmse),
                "corr": float(corr),
                "mean_error": float(np.mean(differences[:, index])),
                "std_error": float(np.std(differences[:, index])),
            }
            print(
                f"{name}: corr={metrics[name]['corr']:.4f}, "
                f"RMSE={metrics[name]['rmse']:.2f}, "
                f"mean error={metrics[name]['mean_error']:.2f}"
            )

        return y_true, y_pred, metrics, differences

    def __call__(self):
        y_pred_scaled, y_true_scaled, embedder_output = self.evaluate()
        y_true, y_pred, metrics, differences = self.compute_metrics(y_pred_scaled, y_true_scaled)
        return y_true, y_pred, metrics, differences, embedder_output
