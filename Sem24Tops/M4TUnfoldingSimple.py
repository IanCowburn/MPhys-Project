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

y_true = np.asarray(model_outputs['y_true'])
y_pred = np.asarray(model_outputs['y_pred'])

bins = [1000,1250,1500,1750,2000,2250,2500]
num_bins = len(bins) - 1

data_true, _ = np.histogram(y_true, bins=bins)
data_pred, _ = np.histogram(y_pred, bins=bins)

fig, ax = plt.subplots()
ax.step(np.arange(num_bins), data_true, where='mid', lw=3,
        alpha=0.7, label='True distribution')
ax.step(np.arange(num_bins), data_pred, where='mid', lw=3,
        alpha=0.7, label='Predicted distribution')
ax.set(xlabel='X bins', ylabel='Counts')
ax.legend()
plt.savefig("simple_unfolding.png")

data_pred_err = np.sqrt(data_pred)
efficiencies = np.ones_like(data_true)
efficiencies_err = np.full_like(data_true, 0.1)

