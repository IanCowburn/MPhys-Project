import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = pd.read_csv('penguins.csv').dropna(inplace = False)

input_features = input_file["flipper_length_mm"].values
target         = input_file["body_mass_g"].values

print(input_features)
print(target)

fig, ax = plt.subplots()

def plot_regression_problem(ax, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
    ax.scatter(input_features, target, marker=".")
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel("Flipper length (mm)")
    ax.set_ylabel("Body mass (g)")

plot_regression_problem(ax)

plt.show()