# Imports
from MPhys4TopInvariantMassTransformerDataLoading import TransformerDataLoader
from MPhys4TopInvariantMassTransformerTraining import TransformerScaling, TransformerTraining
from MPhys4TopInvariantMassTransformerEvaluation import TransformerEvaluation
from MPhys4TopInvariantMassTransformerPlotting import TransformerPlotting


# Data files
files = ["tttt_NLO_523243_mc23a_fullsim.root",
         "tttt_NLO_523243_mc23d_fullsim.root",
         "tttt_NLO_523243_mc23e_fullsim.root"]
# Data list preparation
lepton_mask_size = 5
jet_mask_size = 20
in_features_jets = 6 # eta, phi, pt, e, 0, bjet_tagging
in_features_leptons = 6 # eta, phi, pt, e, lepton_charge, 0
# Relevant variable names
var_names = ["lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "lepton_pt_NOSYS", "jet_pt_NOSYS", "lepton_e_NOSYS", "jet_e_NOSYS", "jet_GN2v01_FixedCutBEff_77_select", "lepton_charge"]
        
loader = TransformerDataLoader(files, var_names, lepton_mask_size, jet_mask_size)
X, y, pad_mask_np = loader()

scaler = TransformerScaling(X, pad_mask_np, y, lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets)
(X_train, X_valid, X_test, y_train, y_valid, y_test, mask_train, mask_valid, mask_test,
valid_tokens_train, valid_tokens_valid, valid_tokens_test, scaler_y,
E_train, T_train, F, E_valid, E_test) = scaler()

input_dim = X_train.shape[2]
embed_dim = 128
n_heads = 4
num_layers = 3
epochs = 250
batch_size = 512

training = TransformerTraining(lepton_mask_size, jet_mask_size, in_features_leptons, in_features_jets,
                 X_train, X_test, X_valid, mask_train, mask_test, mask_valid, y_train, y_test, y_valid,
                 input_dim, embed_dim, n_heads, num_layers, epochs, batch_size)
model, X_test, y_test, mask_test = training()

evaluation = TransformerEvaluation(model, X_test, y_test, mask_test, scaler_y, batch_size)
y_true, y_pred, corr, rmse_biased, rmse, difference, embedder_output = evaluation()

plotting = TransformerPlotting(difference, embedder_output, y_true, y_pred, rmse)
plotting.plot_all()