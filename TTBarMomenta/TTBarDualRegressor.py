from TTBarDualRegressorTraining import TransformerScaling, TransformerTraining
from TTBarDualRegressorEvaluation import TransformerEvaluation
from TTBarDualRegressorPlotting import TransformerPlotting
import pickle
import torch
import onnxruntime as ort
import numpy as np
import h5py


def load_ttbar_h5_for_dual_regressor(filepath="ttbar2L_withEllipse.h5"):
    with h5py.File(filepath, "r") as f:
        inputs = f["INPUTS"]
        targets = f["TARGETS"]

        jets = inputs["jets"][...]
        leptons = inputs["leptons"][...]
        met = inputs["met"][...]
        em_antinu = inputs["EM_antinu"][...]
        em_nu = inputs["EM_nu"][...]

        antinu_target = targets["antinu_target"][...]
        nu_target = targets["nu_target"][...]

    jets_feat = np.stack(
        [
            jets["pt"],
            jets["eta"],
            jets["phi"],
            jets["e"],
            jets["btag77"],
            np.zeros_like(jets["pt"]),
        ],
        axis=-1,
    ).astype(np.float32)

    leptons_feat = np.stack(
        [
            leptons["pt"],
            leptons["eta"],
            leptons["phi"],
            leptons["e"],
            leptons["charge"],
            leptons["type"],
        ],
        axis=-1,
    ).astype(np.float32)

    met_feat = np.stack(
        [
            met["met"],
            met["phi"],
            np.zeros_like(met["met"]),
            np.zeros_like(met["met"]),
            np.zeros_like(met["met"]),
            np.zeros_like(met["met"]),
        ],
        axis=-1,
    )[:, np.newaxis, :].astype(np.float32)

    ellipse_anti_nu_feat = np.stack(
        [
            em_antinu["px"],
            em_antinu["py"],
            em_antinu["pz"],
            em_antinu["e"],
            np.zeros_like(em_antinu["px"]),
            np.zeros_like(em_antinu["px"]),
        ],
        axis=-1,
    )[:, np.newaxis, :].astype(np.float32)

    ellipse_nu_feat = np.stack(
        [
            em_nu["px"],
            em_nu["py"],
            em_nu["pz"],
            em_nu["e"],
            np.zeros_like(em_nu["px"]),
            np.zeros_like(em_nu["px"]),
        ],
        axis=-1,
    )[:, np.newaxis, :].astype(np.float32)

    X = np.concatenate([leptons_feat, jets_feat, met_feat, ellipse_nu_feat, ellipse_anti_nu_feat], axis=1).astype(np.float32)

    y = np.stack([antinu_target["px"], antinu_target["py"], antinu_target["pz"], nu_target["px"], nu_target["py"], nu_target["pz"]], axis=1).astype(np.float32)

    lepton_pad = np.isclose(leptons_feat[:, :, 0], 0.0) & np.isclose(leptons_feat[:, :, 3], 0.0)
    jet_pad = np.isclose(jets_feat[:, :, 0], 0.0) & np.isclose(jets_feat[:, :, 3], 0.0)

    # Only leptons and jets are padded.
    # MET, ellipse_nu, ellipse_anti_nu are single tokens per event and are never masked.
    pad_mask_np = np.concatenate(
        [
            lepton_pad,
            jet_pad,
            np.zeros((X.shape[0], 3), dtype=bool),
        ],
        axis=1,
    )

    return X, y, pad_mask_np


lepton_mask_size = 2
jet_mask_size = 8
in_features_jets = 5
in_features_leptons = 6
in_features_met = 2
in_features_ellipse_nu = 4
in_features_ellipse_anti_nu = 4

X, y, pad_mask_np = load_ttbar_h5_for_dual_regressor("ttbar2L_withEllipse.h5")
# Remove outlier events where any target exceeds 2000 (catastrophic EM failures)
clean = np.max(np.abs(y), axis=1) < 2000
print(f"Outlier cut: keeping {clean.sum()}/{len(y)} events ({100*clean.mean():.2f}%)")
X, y, pad_mask_np = X[clean], y[clean], pad_mask_np[clean]
np.save("ttbar_dual_cache_X.npy", X)
np.save("ttbar_dual_cache_y.npy", y)
np.save("ttbar_dual_cache_pad_mask.npy", pad_mask_np)

scaler = TransformerScaling(X, pad_mask_np, y,
                            lepton_mask_size, jet_mask_size,
                            in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu)
(X_train, X_valid, X_test,
 y_train, y_valid, y_test,
 mask_train, mask_valid, mask_test,
 scaler_y, x_scalers) = scaler()
input_dim = X_train.shape[2]
embed_dim = 256
n_heads = 8
num_layers = 6
epochs = 50
batch_size = 1024
training = TransformerTraining(lepton_mask_size, jet_mask_size,
                               in_features_leptons, in_features_jets, in_features_met, in_features_ellipse_nu, in_features_ellipse_anti_nu,
                               X_train, X_valid, X_test,
                               mask_train, mask_valid, mask_test,
                               y_train, y_valid, y_test,
                               input_dim, embed_dim, n_heads, num_layers,
                               epochs, batch_size, scaler_y, x_scalers)
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
    'in_features_ellipse_nu': in_features_ellipse_nu,
    'in_features_ellipse_anti_nu': in_features_ellipse_anti_nu,
}, 'ttbar_transformer_model_with_ellipse.pt')

evaluation = TransformerEvaluation(model, test_loader, scaler_y, device, batch_size)
y_true, y_pred, metrics, differences, embedder_output = evaluation()

var_ratio = np.var(y_pred) / np.var(y_true)
print(f"Variance Ratio: {var_ratio}")

plotting = TransformerPlotting(y_true, y_pred, metrics, differences, train_loss_history, val_loss_history, suffix="_with_ellipse")
plotting.plot_all()

# Save model outputs for later plotting
model_outputs = {
    'y_true': y_true,
    'y_pred': y_pred,
    'metrics': metrics,
    'differences': differences,
    'scaler_y': scaler_y,
}

with open('ttbar_transformer_model_outputs_with_ellipse.pkl', 'wb') as f:
    pickle.dump(model_outputs, f)

print("Model outputs saved to ttbar_transformer_model_outputs_with_ellipse.pkl")

# Export to ONNX format
model.eval()
token_count = lepton_mask_size + jet_mask_size + 3
dummy_input = torch.randn(1, token_count, input_dim).to(device)
dummy_mask = torch.zeros(1, token_count, dtype=torch.bool).to(device)

torch.onnx.export(
    model,
    (dummy_input, dummy_mask),
    'ttbar_transformer_model_with_ellipse.onnx',
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
print("Model exported to ttbar_transformer_model_with_ellipse.onnx")

# Load and test the ONNX model
ort_session = ort.InferenceSession("ttbar_transformer_model_with_ellipse.onnx")

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