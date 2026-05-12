from M4TLargeRegressorDataLoadingInference import TransformerDataLoader
from M4TLargeRegressorTrainingInference import TransformerScaling, TransformerTraining
from M4TLargeRegressorEvaluationInference import TransformerEvaluation
from M4TLargeRegressorPlottingInference import TransformerPlotting


import pickle
import torch
import onnxruntime as ort

# Data files
files = ["4topLO_24March26_minus.root", "4topLO_24March26_plus.root"]

var_names = ['el_eta', 'el_phi', 'el_pt', 'el_charge',
             'mu_eta', 'mu_phi', 'mu_pt', 'mu_charge',
             'jet_eta', 'jet_phi', 'jet_pt', 'jet_mass', 'jet_btag',
             'met_phi', 'met_met']



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

for suffix in ['_for_report_inference']:

    scaler_save_path = f'four_top_large_transformer{suffix}_scalers.pkl'

    scaler = TransformerScaling(X, pad_mask_np, y,
                                lepton_mask_size, jet_mask_size,
                                in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht)
    (X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    mask_train, mask_valid, mask_test,
    scaler_y) = scaler(save_scalers_path=scaler_save_path)
    input_dim = X_train.shape[2]
    embed_dim = 256
    n_heads = 8
    num_layers = 10
    epochs = 50
    batch_size = 1024
    training = TransformerTraining(lepton_mask_size, jet_mask_size,
                                in_features_leptons, in_features_jets, in_features_met, in_features_numbers, in_features_ht,
                                X_train, X_valid, X_test,
                                mask_train, mask_valid, mask_test,
                                y_train, y_valid, y_test,
                                input_dim, embed_dim, n_heads, num_layers,
                                epochs, batch_size, learning_rate = 4e-4, kl_weight=0, kl_warmup_epochs=15, warmup_epochs=10, num_bins=128, early_stopping_patience=10)
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
        'in_features_ht': in_features_ht,
        'target_log_base': 'natural',
        'target_unit': 'GeV'
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
