# Imports
from FourTopDualRegressorDataLoading import TransformerDataLoader
from FourTopDualRegressorTraining import TransformerScaling, TransformerTraining
from FourTopDualRegressorEvaluation import TransformerEvaluation
from FourTopDualRegressorPlotting import TransformerPlotting
import pickle
import torch

# Data files
files = ["scratch/tttt_NLO_523243_mc23a_fullsim.root",
         "scratch/tttt_NLO_523243_mc23d_fullsim.root",
         "scratch/tttt_NLO_523243_mc23e_fullsim.root"]
# Data list preparation
lepton_mask_size = 2
jet_mask_size = 12
in_features_jets = 6 # eta, phi, pt, e, charge (0 for jets), bjet_tagging
in_features_leptons = 6 # eta, phi, pt, e, lepton_charge, bjet_tag (0 for leptons)
# Note: MET is now a separate 15th token with features [0, met_phi, met_met, 0, 0, 0]
# Relevant variable names
var_names = ["lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "lepton_pt_NOSYS", "jet_pt_NOSYS", "lepton_e_NOSYS", "jet_e_NOSYS", "met_met_NOSYS", "met_phi_NOSYS", "jet_GN2v01_FixedCutBEff_77_select", "lepton_charge"]


loader = TransformerDataLoader(files, var_names, lepton_mask_size, jet_mask_size)
X, y, pad_mask_np = loader()
scaler = TransformerScaling(X, pad_mask_np, y,
                            lepton_mask_size, jet_mask_size,
                            in_features_leptons, in_features_jets)
(X_train, X_valid, X_test,
 y_train, y_valid, y_test,
 mask_train, mask_valid, mask_test,
 scaler_y) = scaler()
input_dim = X_train.shape[2]
embed_dim = 64
n_heads = 8
num_layers = 6
epochs = 250
batch_size = 512
training = TransformerTraining(lepton_mask_size, jet_mask_size,
                               in_features_leptons, in_features_jets,
                               X_train, X_valid, X_test,
                               mask_train, mask_valid, mask_test,
                               y_train, y_valid, y_test,
                               input_dim, embed_dim, n_heads, num_layers,
                               epochs, batch_size)
model, test_loader, device = training()

torch.save({
    'model_state_dict': model.state_dict(),
    'embed_dim': embed_dim,
    'n_heads': n_heads,
    'num_layers': num_layers,
    'lepton_mask_size': lepton_mask_size,
    'jet_mask_size': jet_mask_size,
    'in_features_leptons': in_features_leptons,
    'in_features_jets': in_features_jets
}, 'transformer_model.pt')

evaluation = TransformerEvaluation(model, test_loader, scaler_y, device, batch_size)
y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased, embedder_output = evaluation()
plotting = TransformerPlotting(y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased, embedder_output)
plotting.plot_all()

# Save model outputs for later plotting (in GeV)
model_outputs = {
    'y_true': y_true / 1e3,  # Convert to GeV
    'y_pred': y_pred / 1e3,  # Convert to GeV
    'invariant_mass_corr': invariant_mass_corr,
    'invariant_mass_rmse': invariant_mass_rmse / 1e3,  # Convert to GeV
    'invariant_mass_difference': invariant_mass_difference / 1e3,  # Convert to GeV
    'ht_corr': ht_corr,
    'ht_rmse': ht_rmse / 1e3,  # Convert to GeV
    'ht_difference': ht_difference / 1e3,  # Convert to GeV
    'scaler_y': scaler_y
}

with open('transformer_model_outputs.pkl', 'wb') as f:
    pickle.dump(model_outputs, f)

print("Model outputs saved to transformer_model_outputs.pkl")

# Export to ONNX format
model.eval()
dummy_input = torch.randn(1, 15, 6).to(device)  # (batch_size=1, 15 tokens, 6 features)
dummy_mask = torch.zeros(1, 15, dtype=torch.bool).to(device)

torch.onnx.export(
    model,
    (dummy_input, dummy_mask),
    'transformer_model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input', 'padding_mask'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'padding_mask': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("Model exported to transformer_model.onnx")