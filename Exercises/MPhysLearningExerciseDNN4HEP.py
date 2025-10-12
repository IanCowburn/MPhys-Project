import h5py
import pandas as pd
import numpy as np
import torch
import requests
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

url = "https://cernbox.cern.ch/remote.php/dav/public-files/icjK5HWChdTcdb2/WW_vs_TT_dataset.h5"

response = requests.get(url)
response.raise_for_status()

H_vs_TT_dataset = io.BytesIO(response.content)

file = h5py.File(H_vs_TT_dataset, 'r')

df_signal       = pd.DataFrame(file['Signal'][:])
df_background   = pd.DataFrame(file['Background'][:])

# print(df_signal)
# print(df_background)

# def compare_distributions(signal_data, background_data, variable_name):
#     plt.figure(figsize=(10, 6),dpi=100)
#     plt.hist(signal_data[variable_name], bins=40,  histtype='step', label='Signal', density=True)
#     plt.hist(background_data[variable_name], bins=40, histtype='step', label='Background', density=True)
#     plt.xlabel(variable_name)
#     plt.ylabel('Density')
#     plt.title(f'Distribution of {variable_name}')
#     plt.legend()
#     plt.show()

input_features = ['Njets', 'HT_all', 'combined_leptons_mass', 'angle_between_jets', 'angle_between_leptons']
# for feature in list_of_selected_features:
#     compare_distributions(df_signal, df_background, feature)

df_signal_filtered = df_signal[input_features]
df_background_filtered = df_background[input_features]

y_signal = np.ones(len(df_signal_filtered))
y_background = np.zeros(len(df_background_filtered))

input_data = np.concatenate((df_signal_filtered, df_background_filtered), axis=0)
target = np.concatenate((y_signal, y_background), axis=0)

indices = np.arange(len(input_data))
np.random.shuffle(indices)

shuffled_input_data = input_data[indices]
shuffled_target = target[indices]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    shuffled_input_data, shuffled_target, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=1/9, random_state=42
)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_val   = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
X_test  = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val   = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
y_test  = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(len(input_features), 100),
    nn.Sigmoid(),
    nn.Dropout(0.2),
    nn.Linear(100, 50),
    nn.Sigmoid(),
    nn.Dropout(0.2),
    nn.Linear(50, 1),
    nn.Sigmoid())

loss_fn = nn.BCELoss()   
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

train_losses = []
val_losses = []
num_epochs = 50

class Early_Stopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience        # How many epochs to wait
        self.min_delta = min_delta      # Minimum improvement to count
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = Early_Stopping(patience=10, min_delta=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training loop over batches
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)  # compute loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)  # sum loss over batch

    # Compute average loss over all training samples
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss_batch = loss_fn(val_outputs, val_targets)
            val_loss += val_loss_batch.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

    scheduler.step()

model.eval()  
with torch.no_grad():
    y_pred_prob = model(X_test)

signal_probs = y_pred_prob[y_test[:,0] == 1].numpy()
background_probs = y_pred_prob[y_test[:,0] == 0].numpy()

plt.hist(signal_probs, bins=50, alpha=0.6, label='Signal', color='blue', density=True)
plt.hist(background_probs, bins=50, alpha=0.6, label='Background', color='red', density=True)

plt.xlabel('Predicted Probability of Signal')
plt.ylabel('Density')
plt.title('Signal vs Background Separation on Test Data')
plt.legend()
plt.show()

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

final_prediction_score = y_pred_prob.numpy()
final_prediction = np.round(final_prediction_score)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
accuracy = accuracy_score(y_test, final_prediction)
precision = precision_score(y_test, final_prediction, average='weighted')
recall = recall_score(y_test, final_prediction, average='weighted')
f1 = f1_score(y_test, final_prediction, average='weighted')
fpr, tpr, thresholds = roc_curve(y_test, final_prediction_score)
roc_auc = auc(fpr, tpr)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

def permutation_importance(model, X_val, y_val):

    detatch_to_binary = lambda x: np.round(model(x).detach().numpy())

    baseline_preds = detatch_to_binary(X_val)
    baseline_accuracy = accuracy_score(y_val, baseline_preds)

    importances = {}
    
    for i in range(X_val.shape[1]):
        X_val_permuted = X_val.clone()
        permuted_column = X_val_permuted[:, i][torch.randperm(X_val_permuted.size(0))]
        X_val_permuted[:, i] = permuted_column
        
        permuted_preds = detatch_to_binary(X_val_permuted)
        permuted_accuracy = accuracy_score(y_val, permuted_preds)
        
        importances[input_features[i]] = baseline_accuracy - permuted_accuracy

    return importances

importances = permutation_importance(model, X_val, y_val)
sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
print("Feature importances (from highest to lowest):")
for feature, importance in sorted_importances.items():
    print(f"{feature}: {importance}")

plt.figure(figsize=(10, 6))
plt.bar(sorted_importances.keys(), sorted_importances.values())
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances via Permutation Importance')
plt.show()

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

model.apply(reset_weights)

