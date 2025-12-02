
# Imports
from FourTopDualRegressorDataLoading import TransformerDataLoader
from FourTopDualRegressorTraining import TransformerScaling, TransformerTraining
from FourTopDualRegressorEvaluation import TransformerEvaluation
from FourTopDualRegressorPlotting import TransformerPlotting

# Data files
files = ["tttt_NLO_523243_mc23a_fullsim.root",
         "tttt_NLO_523243_mc23d_fullsim.root",
         "tttt_NLO_523243_mc23e_fullsim.root"]
# Data list preparation
lepton_mask_size = 2
jet_mask_size = 12
in_features_jets = 6 # eta, phi, pt, e, 0, bjet_tagging
in_features_leptons = 6 # eta, phi, pt, e, lepton_charge, 0
# Relevant variable names
var_names = ["lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "lepton_pt_NOSYS", "jet_pt_NOSYS", "lepton_e_NOSYS", "jet_e_NOSYS", "jet_GN2v01_FixedCutBEff_77_select", "lepton_charge"]
        

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
n_heads = 4
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
evaluation = TransformerEvaluation(model, test_loader, scaler_y, device, batch_size)
y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, embedder_output = evaluation()
plotting = TransformerPlotting(y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, embedder_output)
plotting.plot_all()