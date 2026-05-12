import numpy as np
import pickle

with open('four_top_large_transformerfine_tuned_with_atlas_1_outputs.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract true and predicted values
y_true = model_data['y_true']
y_pred = model_data['y_pred']

def kl_divergence_from_samples(true_samples, pred_samples, bins=500, range_min=None, range_max=None):
    """Calculate KL divergence between two distributions estimated from samples."""
    if range_min is None or range_max is None:
        range_min = min(true_samples.min(), pred_samples.min())
        range_max = max(true_samples.max(), pred_samples.max())
    
    counts_true, _ = np.histogram(true_samples, bins=bins, range=(range_min, range_max))
    counts_pred, _ = np.histogram(pred_samples, bins=bins, range=(range_min, range_max))
    
    # Normalize to probabilities
    p = counts_true / counts_true.sum()
    q = counts_pred / counts_pred.sum()
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # KL(p || q) = sum(p * log(p / q))
    kl = np.sum(p * (np.log(p) - np.log(q)))
    return kl

# Calculate KL divergence for both distributions
# y_true = y_true * 1e3  # Convert back to GeV
# y_pred = y_pred * 1e3  # Convert back to GeV

print(y_true[:,0].min(), y_true[:,0].max())
print(y_pred[:,0].min(), y_pred[:,0].max())
print(y_true[:,1].min(), y_true[:,1].max())
print(y_pred[:,1].min(), y_pred[:,1].max())

kl_mass = kl_divergence_from_samples(y_true[:, 0], y_pred[:, 0], bins=500, range_min=0, range_max=6e3)
kl_ht = kl_divergence_from_samples(y_true[:, 1], y_pred[:, 1], bins=500, range_min=0, range_max=4e3)



print(f"\nKL Divergence (truth || prediction):")
print(f"  Invariant Mass: {kl_mass:.6f}")
print(f"  $H_T$: {kl_ht:.6f}\n")