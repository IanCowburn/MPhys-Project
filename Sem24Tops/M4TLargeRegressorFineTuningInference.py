from M4TLargeRegressorFineTuningDataLoadingInference import TransformerDataLoader
from M4TLargeRegressorTrainingInference import TransformerScaling, TransformerTraining
from M4TLargeRegressorEvaluationInference import TransformerEvaluation
from M4TLargeRegressorPlottingInference import TransformerPlotting


import os
import numpy as np
import torch
import onnxruntime as ort
import pickle

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




loader = TransformerDataLoader(files, var_names, lepton_mask_size, jet_mask_size)
X, y, pad_mask_np = loader()

pretrained_scaler_path = "four_top_large_transformer_no_kl_warmup_15_128_bins_50_epochs_early_5_256_8_10_1024_dim_feedforward_scalers.pkl"
pretrained_model_path = "four_top_large_transformer_no_kl_warmup_15_128_bins_50_epochs_early_5_256_8_10_1024_dim_feedforward.pt" 
fine_tune_learning_rate = 1e-5


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

def check_target_vs_pretrained_scaler(y_np, scaler_y):
    y_mass = y_np[:, 0] if y_np.ndim == 2 else y_np
    y_mass = np.clip(y_mass, 1e-12, None)

    med_log = float(np.nanmedian(np.log(y_mass)))
    scaler_mu = float(np.ravel(scaler_y.mean_)[0])
    delta = med_log - scaler_mu
    print(f"[UnitCheck] median(log(y_mass))={med_log:.3f}, scaler_mean={scaler_mu:.3f}, delta={delta:.3f}")

    if abs(delta - np.log(1e3)) < 1.0:
        print("[UnitCheck][WARNING] y still looks ~1e3 high vs pretrained scaler.")

def transform_targets_with_loaded_scaler(y_tensor, fitted_target_scaler):
    y_np = _to_numpy(y_tensor).astype(np.float32)
    y_np = np.clip(y_np, 1e-12, None)
    y_log = np.log(y_np).reshape(-1, 1)
    y_scaled = fitted_target_scaler.transform(y_log).reshape(y_np.shape)
    return torch.from_numpy(y_scaled).float()

def split_dataset_two_way(X_np, y_np, mask_np, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X_np.shape[0])
    mid = X_np.shape[0] // 2
    idx_a = indices[:mid]
    idx_b = indices[mid:]
    return (X_np[idx_a], y_np[idx_a], mask_np[idx_a]), (X_np[idx_b], y_np[idx_b], mask_np[idx_b])

# Critical fix: align ATLAS units to pretrained GeV convention
X, y = maybe_convert_atlas_mev_to_gev(X, y, lepton_mask_size, jet_mask_size)

split_a, split_b = split_dataset_two_way(X, y, pad_mask_np, seed=42)
splits = [
    ("splitA", split_a),
    ("splitB", split_b)
]

if not os.path.exists(pretrained_scaler_path):
    raise FileNotFoundError(
        f"Could not find saved scaler bundle at {pretrained_scaler_path}. "
        "Run base training first or update pretrained_scaler_path."
    )

if not os.path.exists(pretrained_model_path):
    raise FileNotFoundError(
        f"Could not find pretrained model checkpoint at {pretrained_model_path}. "
        "Set pretrained_model_path to your base .pt file."
    )

base_ckpt = torch.load(pretrained_model_path, map_location="cpu")
pretrained_state_dict = base_ckpt["model_state_dict"]

scaler_bundle = TransformerScaling.load_scalers(pretrained_scaler_path)
feature_scalers = scaler_bundle["feature_scalers"]
scaler_y = scaler_bundle["target_scaler"]

layout = scaler_bundle.get("layout", {})
if layout and (
    layout.get("lepton_mask_size") != lepton_mask_size
    or layout.get("jet_mask_size") != jet_mask_size
):
    raise ValueError(
        "Loaded scaler layout does not match this dataset token layout. "
        f"Loaded lepton/jet mask sizes: {layout.get('lepton_mask_size')}/{layout.get('jet_mask_size')}, "
        f"current: {lepton_mask_size}/{jet_mask_size}."
    )

for split_label, (X_split, y_split, mask_split) in splits:
    suffix = f"_for_report_fine_tuned_inference_{split_label}"

    scaler = TransformerScaling(X_split, mask_split, y_split,
                                lepton_mask_size, jet_mask_size,
                                in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht)

    check_target_vs_pretrained_scaler(y_split, scaler_y)

    (X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    mask_train, mask_valid, mask_test,
    _, _, _) = scaler.prepare_data(X_split, y_split, mask_split)

    y_train = transform_targets_with_loaded_scaler(y_train, scaler_y)
    y_valid = transform_targets_with_loaded_scaler(y_valid, scaler_y)
    y_test = transform_targets_with_loaded_scaler(y_test, scaler_y)

    X_train = scaler.scale_X(X_train, mask_train, feature_scalers)
    X_valid = scaler.scale_X(X_valid, mask_valid, feature_scalers)
    X_test = scaler.scale_X(X_test, mask_test, feature_scalers)

    input_dim = X_train.shape[2]

    # Use architecture from checkpoint (fallbacks kept)
    embed_dim = base_ckpt.get("embed_dim", 256)
    n_heads = base_ckpt.get("n_heads", 8)
    num_layers = base_ckpt.get("num_layers", 10)

    epochs = 50
    batch_size = 1024
    training = TransformerTraining(lepton_mask_size, jet_mask_size,
                                in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht,
                                X_train, X_valid, X_test,
                                mask_train, mask_valid, mask_test,
                                y_train, y_valid, y_test,
                                input_dim, embed_dim, n_heads, num_layers,
                                epochs, batch_size,
                                learning_rate=fine_tune_learning_rate,
                                pretrained_state_dict=pretrained_state_dict)  # <-- add this
    model, test_loader, device, train_loss_history, val_loss_history = training()

    torch.save({
        'model_state_dict': model.state_dict(),
        'embed_dim': embed_dim,
        'n_heads': n_heads,
        'num_layers': num_layers,
        'lepton_mask_size': lepton_mask_size,
        'jet_mask_size': jet_mask_size,
        'in_features_leptons': in_features_leptons,
        'in_features_jets': in_features_jets,
        'in_features_met': in_features_met,
        'in_features_numbers': in_features_numbers,
        'in_features_ht': in_features_ht
    }, f'four_top_large_transformer{suffix}.pt')

    evaluation = TransformerEvaluation(model, test_loader, scaler_y, device, batch_size)
    y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased, embedder_output = evaluation()
    plotting = TransformerPlotting(y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased, embedder_output, train_loss_history, val_loss_history, suffix)
    plotting.plot_all()

    # Save model outputs for later plotting (already in GeV)
    model_outputs = {
        'y_true': y_true,
        'y_pred': y_pred,
        'invariant_mass_corr': invariant_mass_corr,
        'invariant_mass_rmse': invariant_mass_rmse,
        'invariant_mass_difference': invariant_mass_difference,
        'ht_corr': ht_corr,
        'ht_rmse': ht_rmse,
        'ht_difference': ht_difference,
        'scaler_y': scaler_y
    }

    with open(f'four_top_large_transformer{suffix}_outputs.pkl', 'wb') as f:
        pickle.dump(model_outputs, f)

    print(f"Model outputs saved to four_top_large_transformer{suffix}_outputs.pkl")

    # Export to ONNX format
    model.eval()
    dummy_input = torch.randn(1, 17, 5).to(device)  # (batch_size=1, 17 tokens, 5 features)
    dummy_mask = torch.zeros(1, 17, dtype=torch.bool).to(device)

    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        f'four_top_large_transformer{suffix}.onnx',
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input', 'padding_mask'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'padding_mask': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to four_top_large_transformer{suffix}.onnx")

    # Load and test the ONNX model
    ort_session = ort.InferenceSession(f"four_top_large_transformer{suffix}.onnx")

    # Prepare inputs for ONNX model
    onnx_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy(),
                ort_session.get_inputs()[1].name: dummy_mask.cpu().numpy()}

    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)
    with torch.no_grad():
        original_output = model(dummy_input, dummy_mask)
    matches = torch.allclose(torch.tensor(onnx_outputs[0]), original_output.cpu(), rtol=1e-03, atol=1e-05)
    if matches:
        print("ONNX model output matches original PyTorch model output.")
    else:
        print("ONNX model output does NOT match original PyTorch model output.")
