from M4TLargeRegressorFineTuningDataLoading import TransformerDataLoader
from M4TLargeRegressorTraining import TransformerScaling
from M4TLargeRegressorPlotting import TransformerPlotting

import os
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

# Data files
files = ["tttt_NLO_523243_mc23a_fullsim.root",
         "tttt_NLO_523243_mc23d_fullsim.root",
         "tttt_NLO_523243_mc23e_fullsim.root"]
         
var_names = ["lepton_eta", "lepton_phi", "lepton_pt_NOSYS", "lepton_e_NOSYS", "lepton_charge",
             "jet_eta", "jet_phi", "jet_pt_NOSYS", "jet_e_NOSYS", "jet_GN2v01_FixedCutBEff_77_select",
             "met_phi_NOSYS", "met_met_NOSYS",
             "nJets", "nBjets_GN2v01_77WP", "nElectrons", "nMuons",
             "HT_all_NOSYS", "HT_jets_NOSYS"]

# Data list preparation
lepton_mask_size = 2
jet_mask_size = 12
in_features_jets = 5 # eta, phi, pt, e, bjet_tagging
in_features_leptons = 5 # eta, phi, pt, e, lepton_charge
in_features_met = 2 # met_phi, met_met
in_features_numbers = 4 # nJets, nBjets, nElectrons, nMuons
in_features_ht = 2 # HT_all, HT_jets

# Most recent fine-tuning output
outputs_path = "four_top_large_transformer_for_report_fine_tuned_outputs.pkl"
suffix = "_for_report_fine_tuned_8plus"


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def maybe_convert_atlas_mev_to_gev(X, y, lepton_mask_size, jet_mask_size):
    X_np = _to_numpy(X).astype(np.float32, copy=True)
    y_np = _to_numpy(y).astype(np.float32, copy=True)

    y_mass = y_np[:, 0] if y_np.ndim == 2 else y_np
    y_ht = y_np[:, 1] if (y_np.ndim == 2 and y_np.shape[1] > 1) else y_mass

    med_mass = float(np.nanmedian(y_mass))
    med_ht = float(np.nanmedian(y_ht))

    # GeV targets are O(1e3); MeV targets are O(1e6)
    needs_mev_to_gev = (med_mass > 2e4) or (med_ht > 1e4)
    if not needs_mev_to_gev:
        print(f"[UnitFix] No conversion applied (median mass={med_mass:.1f}, HT={med_ht:.1f}).")
        return X_np, y_np

    s = 1e-3  # MeV -> GeV
    lep = slice(0, lepton_mask_size)
    jet = slice(lepton_mask_size, lepton_mask_size + jet_mask_size)
    met_idx = lepton_mask_size + jet_mask_size
    ht_idx = met_idx + 2  # tokens: [met, numbers, ht]

    # energy-like features
    X_np[:, lep, 2] *= s      # lepton pt
    X_np[:, lep, 3] *= s      # lepton e
    X_np[:, jet, 2] *= s      # jet pt
    X_np[:, jet, 3] *= s      # jet e
    X_np[:, met_idx, 1] *= s  # met_met
    X_np[:, ht_idx, 0:2] *= s # HT_all, HT_jets

    # targets [invariant_mass, HT]
    y_np *= s

    new_med_mass = float(np.nanmedian(y_np[:, 0] if y_np.ndim == 2 else y_np))
    print(f"[UnitFix] Applied MeV->GeV conversion (median mass {med_mass:.1f} -> {new_med_mass:.1f}).")
    return X_np, y_np


def compute_metrics_from_physical(y_true, y_pred):
    invariant_mass_difference = y_pred[:, 0] - y_true[:, 0]
    ht_difference = y_pred[:, 1] - y_true[:, 1]

    err_counts_mass, err_bins_mass = np.histogram(invariant_mass_difference, bins=200, range=(-1e3, 1e3))
    mode_error_mass = 0.5 * (err_bins_mass[:-1] + err_bins_mass[1:])[np.argmax(err_counts_mass)]
    err_counts_ht, err_bins_ht = np.histogram(ht_difference, bins=200, range=(-1e3, 1e3))
    mode_error_ht = 0.5 * (err_bins_ht[:-1] + err_bins_ht[1:])[np.argmax(err_counts_ht)]

    invariant_mass_biased = y_pred[:, 0] - mode_error_mass
    ht_biased = y_pred[:, 1] - mode_error_ht

    invariant_mass_rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    ht_rmse = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    invariant_mass_corr = np.corrcoef(y_true[:, 0], y_pred[:, 0])[0, 1]
    ht_corr = np.corrcoef(y_true[:, 1], y_pred[:, 1])[0, 1]

    return (invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference,
            ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased)


if not os.path.exists(outputs_path):
    raise FileNotFoundError(
        f"Could not find outputs file at {outputs_path}. "
        "Run fine-tuning first or update outputs_path."
    )

with open(outputs_path, "rb") as f:
    outputs = pickle.load(f)

y_true_full = np.asarray(outputs["y_true"])
y_pred_full = np.asarray(outputs["y_pred"])

loader = TransformerDataLoader(files, var_names, lepton_mask_size, jet_mask_size)
X, y, pad_mask_np = loader()

# Align units and split deterministically to match test set ordering
X, y = maybe_convert_atlas_mev_to_gev(X, y, lepton_mask_size, jet_mask_size)
scaler = TransformerScaling(X, pad_mask_np, y,
                            lepton_mask_size, jet_mask_size,
                            in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht)

(_, _, X_test,
 _, _, _,
 _, _, _,
 _, _, _) = scaler.prepare_data(X, y, pad_mask_np)

numbers_idx = lepton_mask_size + jet_mask_size + 1
n_jets_test = X_test[:, numbers_idx, 0].numpy()
jet_mask = n_jets_test >= 8

if y_true_full.shape[0] != X_test.shape[0]:
    raise ValueError(
        "Test-set size mismatch between outputs and current data split. "
        f"outputs={y_true_full.shape[0]}, split={X_test.shape[0]}"
    )

y_true = y_true_full[jet_mask]
y_pred = y_pred_full[jet_mask]

print(f"8+ jets test events: {y_true.shape[0]} / {y_true_full.shape[0]}")

(invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference,
 ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased) = compute_metrics_from_physical(y_true, y_pred)

plotting = TransformerPlotting(
    y_true, y_pred,
    invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference,
    ht_corr, ht_rmse, ht_difference,
    invariant_mass_biased, ht_biased,
    embedder_output=None,
    train_loss_history=None,
    val_loss_history=None,
    suffix=suffix
)
plotting.plot_all()
