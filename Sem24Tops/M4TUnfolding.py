from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors

with open("four_top_large_transformerfine_tuned_with_atlas_no_kl_outputs.pkl", "rb") as f:
    model_outputs = pickle.load(f)


def equal_count_edges(values, n_bins, value_range):
    """Build variable-width edges so each truth bin has ~equal event counts."""
    low, high = value_range
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= low) & (vals <= high)]

    if vals.size < 2:
        return np.linspace(low, high, n_bins + 1)

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(vals, q)
    edges[0] = low
    edges[-1] = high

    # Keep edges strictly increasing for histogram2d in case of repeated quantiles.
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    return edges

y_true = np.asarray(model_outputs['y_true'])
y_pred = np.asarray(model_outputs['y_pred'])

if y_true.shape != y_pred.shape:
    raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

# Closure-test split: train sample builds response, test sample is unfolded.
rng = np.random.default_rng(42)
n_events = y_true.shape[0]
perm = rng.permutation(n_events)
split_idx = int(0.9 * n_events)
train_idx = perm[:split_idx]
test_idx = perm[split_idx:]

print(f"Response-train events: {train_idx.size}")
print(f"Unfold-test events:    {test_idx.size}")

m4t_true_train = y_true[train_idx, 0]
m4t_pred_train = y_pred[train_idx, 0]
ht_true_train = y_true[train_idx, 1]
ht_pred_train = y_pred[train_idx, 1]

m4t_true_test = y_true[test_idx, 0]
m4t_pred_test = y_pred[test_idx, 0]
ht_true_test = y_true[test_idx, 1]
ht_pred_test = y_pred[test_idx, 1]

n_column_bins = 10

# Equal-statistics binning for columns (truth axis), as requested.
m4t_true_edges = equal_count_edges(m4t_true_train, n_column_bins, (0.0, 6000.0))
ht_true_edges = equal_count_edges(ht_true_train, n_column_bins, (0.0, 4000.0))

# Use the same bin edges on both axes for each variable.
m4t_pred_edges = m4t_true_edges.copy()
ht_pred_edges = ht_true_edges.copy()

m4t_true_hist, _ = np.histogram(m4t_true_train, bins=m4t_true_edges)
m4t_pred_hist, _ = np.histogram(m4t_pred_train, bins=m4t_pred_edges)
ht_true_hist, _ = np.histogram(ht_true_train, bins=ht_true_edges)
ht_pred_hist, _ = np.histogram(ht_pred_train, bins=ht_pred_edges)

m4t_pred_error = m4t_pred_hist / np.sqrt(m4t_pred_hist + 1e-6)
ht_pred_error = ht_pred_hist / np.sqrt(ht_pred_hist + 1e-6)

m4t_efficiencies = np.ones_like(m4t_true_hist)  # Assuming perfect efficiency for simplicity
ht_efficiencies = np.ones_like(ht_true_hist)  # Assuming perfect efficiency for simplicity
m4t_efficiencies_error = np.full_like(m4t_efficiencies, 0.1)  # 10% uncertainty on efficiency
ht_efficiencies_error = np.full_like(ht_efficiencies, 0.1)  # 10% uncertainty on efficiency

# Axis convention: x = truth, y = reconstructed/predicted.
response_matrix_m4t, m4t_true_edges, m4t_pred_edges = np.histogram2d(
    m4t_true_train, m4t_pred_train, bins=[m4t_true_edges, m4t_pred_edges]
)
response_matrix_ht, ht_true_edges, ht_pred_edges = np.histogram2d(
    ht_true_train, ht_pred_train, bins=[ht_true_edges, ht_pred_edges]
)

m4t_response_matrix_error = np.sqrt(response_matrix_m4t + 1e-6)
ht_response_matrix_error = np.sqrt(response_matrix_ht + 1e-6)

suffix = "fine_tuned_with_atlas_no_kl_outputs_model_unfolding_10_bins"

# --- Response Matrix Plots ---

max_count = max(response_matrix_m4t.max(), response_matrix_ht.max())
norm_shared = mcolors.Normalize(vmin=0, vmax=max_count if max_count > 0 else 1)

with plt.rc_context({'font.size': 20}):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    h_mass = axes[0].pcolormesh(
        m4t_true_edges,
        m4t_pred_edges,
        response_matrix_m4t.T,
        cmap='hot',
        norm=norm_shared,
        shading='auto'
    )
    axes[0].plot([0, 6e3], [0, 6e3], color='red', linestyle='--', linewidth=1, label='Perfect prediction')
    axes[0].set_xlabel('True Invariant Mass [GeV]')
    axes[0].set_ylabel('Predicted Invariant Mass [GeV]')
    axes[0].set_xlim(0, 6e3)
    axes[0].set_ylim(0, 6e3)

    h_ht = axes[1].pcolormesh(
        ht_true_edges,
        ht_pred_edges,
        response_matrix_ht.T,
        cmap='hot',
        norm=norm_shared,
        shading='auto'
    )
    axes[1].plot([0, 4e3], [0, 4e3], color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('True $H_T$ [GeV]')
    axes[1].set_ylabel('Predicted $H_T$ [GeV]')
    axes[1].set_xlim(0, 4e3)
    axes[1].set_ylim(0, 4e3)

    cbar = fig.colorbar(h_mass, ax=axes, label='Counts')
    cbar.ax.tick_params(labelsize=18)

    handles_2d = axes[0].get_legend_handles_labels()
    if handles_2d[0]:
        fig.legend(handles_2d[0], handles_2d[1], loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=18)

    fig.savefig(f'four_top_large_transformer{suffix}_response_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# Normalise by truth bins (matrix columns in the plotted, transposed view).
m4t_truth_bin_sums = response_matrix_m4t.sum(axis=1)
ht_truth_bin_sums = response_matrix_ht.sum(axis=1)

m4t_normalisation_factor = m4t_efficiencies / (m4t_truth_bin_sums + 1e-6)
ht_normalisation_factor = ht_efficiencies / (ht_truth_bin_sums + 1e-6)

m4t_response_matrix_normalised = response_matrix_m4t * m4t_normalisation_factor[:, np.newaxis]
ht_response_matrix_normalised = response_matrix_ht * ht_normalisation_factor[:, np.newaxis]

m4t_response_matrix_error_normalised = m4t_response_matrix_error * m4t_normalisation_factor[:, np.newaxis]
ht_response_matrix_error_normalised = ht_response_matrix_error * ht_normalisation_factor[:, np.newaxis]

print("m4t truth-bin sums after normalisation:", m4t_response_matrix_normalised.sum(axis=1))
print("ht truth-bin sums after normalisation:", ht_response_matrix_normalised.sum(axis=1))

norm_shared_normalised = mcolors.Normalize(
    vmin=0,
    vmax=max(m4t_response_matrix_normalised.max(), ht_response_matrix_normalised.max(), 1e-12)
)

with plt.rc_context({'font.size': 20}):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    h_mass_norm = axes[0].pcolormesh(
        m4t_true_edges,
        m4t_pred_edges,
        m4t_response_matrix_normalised.T,
        cmap='hot',
        norm=norm_shared_normalised,
        shading='auto'
    )
    axes[0].plot([0, 6e3], [0, 6e3], color='red', linestyle='--', linewidth=1, label='Perfect prediction')
    axes[0].set_xlabel('True Invariant Mass [GeV]')
    axes[0].set_ylabel('Predicted Invariant Mass [GeV]')
    axes[0].set_xlim(0, 6e3)
    axes[0].set_ylim(0, 6e3)

    h_ht_norm = axes[1].pcolormesh(
        ht_true_edges,
        ht_pred_edges,
        ht_response_matrix_normalised.T,
        cmap='hot',
        norm=norm_shared_normalised,
        shading='auto'
    )
    axes[1].plot([0, 4e3], [0, 4e3], color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('True $H_T$ [GeV]')
    axes[1].set_ylabel('Predicted $H_T$ [GeV]')
    axes[1].set_xlim(0, 4e3)
    axes[1].set_ylim(0, 4e3)

    cbar = fig.colorbar(h_mass_norm, ax=axes, label='Normalised Counts')
    cbar.ax.tick_params(labelsize=18)

    handles_2d = axes[0].get_legend_handles_labels()
    if handles_2d[0]:
        fig.legend(handles_2d[0], handles_2d[1], loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=18)

    fig.savefig(f'four_top_large_transformer{suffix}_normalised_response_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# --- Unfolding setup (coarser bins + reco->truth response orientation) ---
n_unfold_bins = 10

m4t_truth_edges_unfold = equal_count_edges(m4t_true_train, n_unfold_bins, (0.0, 6000.0))
ht_truth_edges_unfold = equal_count_edges(ht_true_train, n_unfold_bins, (0.0, 4000.0))
m4t_reco_edges_unfold = m4t_truth_edges_unfold.copy()
ht_reco_edges_unfold = ht_truth_edges_unfold.copy()

m4t_true_hist_unfold_train, _ = np.histogram(m4t_true_train, bins=m4t_truth_edges_unfold)
m4t_pred_hist_unfold_test, _ = np.histogram(m4t_pred_test, bins=m4t_reco_edges_unfold)
m4t_true_hist_unfold_test, _ = np.histogram(m4t_true_test, bins=m4t_truth_edges_unfold)

ht_true_hist_unfold_train, _ = np.histogram(ht_true_train, bins=ht_truth_edges_unfold)
ht_pred_hist_unfold_test, _ = np.histogram(ht_pred_test, bins=ht_reco_edges_unfold)
ht_true_hist_unfold_test, _ = np.histogram(ht_true_test, bins=ht_truth_edges_unfold)

m4t_pred_error_unfold = np.sqrt(m4t_pred_hist_unfold_test + 1e-6)
ht_pred_error_unfold = np.sqrt(ht_pred_hist_unfold_test + 1e-6)

# For pyunfold: response axis order is effect(reco), cause(truth).
response_matrix_m4t_unfold, m4t_reco_edges_unfold, m4t_truth_edges_unfold = np.histogram2d(
    m4t_pred_train, m4t_true_train, bins=[m4t_reco_edges_unfold, m4t_truth_edges_unfold]
)
response_matrix_ht_unfold, ht_reco_edges_unfold, ht_truth_edges_unfold = np.histogram2d(
    ht_pred_train, ht_true_train, bins=[ht_reco_edges_unfold, ht_truth_edges_unfold]
)

m4t_response_matrix_error_unfold = np.sqrt(response_matrix_m4t_unfold + 1e-6)
ht_response_matrix_error_unfold = np.sqrt(response_matrix_ht_unfold + 1e-6)

m4t_eff_num = response_matrix_m4t_unfold.sum(axis=0)
ht_eff_num = response_matrix_ht_unfold.sum(axis=0)
m4t_eff_den = m4t_true_hist_unfold_train + 1e-12
ht_eff_den = ht_true_hist_unfold_train + 1e-12

# Use a smoothed binomial estimate so efficiencies and their errors are never exactly 0 or 1.
# This avoids singular covariance pieces in pyunfold's internal error propagation.
m4t_efficiencies_unfold = np.clip((m4t_eff_num + 0.5) / (m4t_eff_den + 1.0), 1e-6, 1.0 - 1e-6)
ht_efficiencies_unfold = np.clip((ht_eff_num + 0.5) / (ht_eff_den + 1.0), 1e-6, 1.0 - 1e-6)

m4t_eff_var = m4t_efficiencies_unfold * (1.0 - m4t_efficiencies_unfold) / (m4t_eff_den + 2.0)
ht_eff_var = ht_efficiencies_unfold * (1.0 - ht_efficiencies_unfold) / (ht_eff_den + 2.0)

# Guard against underflow to exactly zero, which can destabilize NCmc in pyunfold.
m4t_efficiencies_error_unfold = np.sqrt(np.maximum(m4t_eff_var, 1e-12))
ht_efficiencies_error_unfold = np.sqrt(np.maximum(ht_eff_var, 1e-12))

unfolded_m4t_results = iterative_unfold(
    data=m4t_pred_hist_unfold_test,
    data_err=m4t_pred_error_unfold,
    response=response_matrix_m4t_unfold,
    response_err=m4t_response_matrix_error_unfold,
    efficiencies=m4t_efficiencies_unfold,
    efficiencies_err=m4t_efficiencies_error_unfold,
    callbacks=[Logger()])

unfolded_ht_results = iterative_unfold(
    data=ht_pred_hist_unfold_test,
    data_err=ht_pred_error_unfold,
    response=response_matrix_ht_unfold,
    response_err=ht_response_matrix_error_unfold,
    efficiencies=ht_efficiencies_unfold,
    efficiencies_err=ht_efficiencies_error_unfold,
    callbacks=[Logger()])

print("Unfolded M4T statistical errors:", unfolded_m4t_results.get("stat_err"))
print("Unfolded HT statistical errors:", unfolded_ht_results.get("stat_err"))
print("Unfolded M4T systematic errors:", unfolded_m4t_results.get("sys_err"))
print("Unfolded HT systematic errors:", unfolded_ht_results.get("sys_err"))

def total_unfold_error(unfold_result):
    stat_err = unfold_result.get("stat_err")
    stat_err = np.asarray(stat_err, dtype=float)
    return stat_err

print("Total unfolded M4T errors:", total_unfold_error(unfolded_m4t_results))
print("Total unfolded HT errors:", total_unfold_error(unfolded_ht_results))


m4t_unfolded = np.asarray(unfolded_m4t_results["unfolded"], dtype=float)
ht_unfolded = np.asarray(unfolded_ht_results["unfolded"], dtype=float)
m4t_unfolded_err = total_unfold_error(unfolded_m4t_results)
ht_unfolded_err = total_unfold_error(unfolded_ht_results)

m4t_bin_centers = 0.5 * (m4t_truth_edges_unfold[:-1] + m4t_truth_edges_unfold[1:])
ht_bin_centers = 0.5 * (ht_truth_edges_unfold[:-1] + ht_truth_edges_unfold[1:])
m4t_bin_widths = np.maximum(np.diff(m4t_truth_edges_unfold), 1e-12)
ht_bin_widths = np.maximum(np.diff(ht_truth_edges_unfold), 1e-12)

m4t_pre_err = np.sqrt(m4t_pred_hist_unfold_test + 1e-6)
ht_pre_err = np.sqrt(ht_pred_hist_unfold_test + 1e-6)

# For variable-width bins, compare dN/dx (not raw counts) to avoid axis-shape artifacts.
m4t_true_density = m4t_true_hist_unfold_test / m4t_bin_widths
m4t_pre_density = m4t_pred_hist_unfold_test / m4t_bin_widths
m4t_unfold_density = m4t_unfolded / m4t_bin_widths
m4t_pre_density_err = m4t_pre_err / m4t_bin_widths
m4t_unfold_density_err = m4t_unfolded_err / m4t_bin_widths

ht_true_density = ht_true_hist_unfold_test / ht_bin_widths
ht_pre_density = ht_pred_hist_unfold_test / ht_bin_widths
ht_unfold_density = ht_unfolded / ht_bin_widths
ht_pre_density_err = ht_pre_err / ht_bin_widths
ht_unfold_density_err = ht_unfolded_err / ht_bin_widths

print("Pre and post unfolding errors per histogram bin: ")
print("M4T Pre-unfold error:", m4t_pre_density_err)
print("M4T Post-unfold error:", m4t_unfold_density_err)
print("HT Pre-unfold error:", ht_pre_density_err)
print("HT Post-unfold error:", ht_unfold_density_err)

with plt.rc_context({'font.size': 16}):
    fig, (ax_mass, ax_ht) = plt.subplots(1, 2, figsize=(16, 6), sharey=False, constrained_layout=True)

    ax_mass.stairs(m4t_true_density, m4t_truth_edges_unfold, color='black', linewidth=1.6, alpha=0.8, label='Truth')
    ax_mass.fill_between(
        m4t_bin_centers, m4t_pre_density - m4t_pre_density_err, m4t_pre_density + m4t_pre_density_err,
        fmt='o', markersize=3, color='tab:blue', ecolor='tab:blue', elinewidth=1, capsize=2,
        alpha=0.8, label='Pre-unfold (predicted)'
    )
    ax_mass.fill_between(
        m4t_bin_centers, m4t_unfold_density - m4t_unfold_density_err, m4t_unfold_density + m4t_unfold_density_err,
        fmt='o', markersize=3, color='tab:red', ecolor='tab:red', elinewidth=1, capsize=2,
        alpha=0.85, label='Post-unfold'
    )
    ax_mass.set_xlabel('Invariant Mass [GeV]')
    ax_mass.set_ylabel('dN/dm [1/GeV]')
    ax_mass.set_xlim(m4t_truth_edges_unfold[0], m4t_truth_edges_unfold[-1])
    ax_mass.grid(True, alpha=0.3)
    ax_mass.legend()

    ax_ht.stairs(ht_true_density, ht_truth_edges_unfold, color='black', linewidth=1.6, alpha=0.8, label='Truth')
    ax_ht.fill_between(
        ht_bin_centers, ht_pre_density - ht_pre_density_err, ht_pre_density + ht_pre_density_err,
        fmt='o', markersize=3, color='tab:blue', ecolor='tab:blue', elinewidth=1, capsize=2,
        alpha=0.8, label='Pre-unfold (predicted)'
    )
    ax_ht.fill_between(
        ht_bin_centers, ht_unfold_density - ht_unfold_density_err, ht_unfold_density + ht_unfold_density_err,
        fmt='o', markersize=3, color='tab:red', ecolor='tab:red', elinewidth=1, capsize=2,
        alpha=0.85, label='Post-unfold'
    )
    ax_ht.set_xlabel('$H_T$ [GeV]')
    ax_ht.set_ylabel('dN/d$H_T$ [1/GeV]')
    ax_ht.set_xlim(ht_truth_edges_unfold[0], ht_truth_edges_unfold[-1])
    ax_ht.grid(True, alpha=0.3)
    ax_ht.legend()

    fig.savefig(
        f'four_top_large_transformer{suffix}_pre_post_unfold_1d.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)

