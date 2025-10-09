import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

input_penguins_df = pd.read_csv("Exercises/penguins.csv")
penguins_df = input_penguins_df.dropna(inplace=False)

fig, ax = plt.subplots()

def plot_categorical_problem(ax, xlow=29, xhigh=61, ylow=12, yhigh=22):
    # Create 3 separate dataframes, one for each species. We use the target array as a mask
    df_Adelie = penguins_df[penguins_df["species"] == "Adelie"]
    df_Gentoo = penguins_df[penguins_df["species"] == "Gentoo"]
    df_Chinstrap = penguins_df[penguins_df["species"] == "Chinstrap"]

    # Plot each species separately with individual colour
    ax.scatter(df_Adelie["bill_length_mm"], df_Adelie["bill_depth_mm"], color="tab:blue", label="Adelie")
    ax.scatter(df_Gentoo["bill_length_mm"], df_Gentoo["bill_depth_mm"], color="tab:orange", label="Gentoo")
    ax.scatter(df_Chinstrap["bill_length_mm"], df_Chinstrap["bill_depth_mm"], color="tab:green", label="Chinstrap")

    # Set plot parameters - less important
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")

    ax.legend(loc="lower left", framealpha=1)

plot_categorical_problem(ax)

plt.show()

penguins_df_no_gentoo = penguins_df[penguins_df["species"] != "Gentoo"]
target, species_names = pd.factorize(penguins_df_no_gentoo["species"])

X = penguins_df_no_gentoo[["bill_length_mm", "bill_depth_mm"]].values
y_true = target

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
best_fit = model.fit(X, y_true)

random_datapoint_features = X[60].reshape(-1,2) # It needs to be reshaped because it's a single sample
random_datapoint_probabilities = best_fit.predict_proba(random_datapoint_features)
print(random_datapoint_probabilities)