import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

input_file = pd.read_csv('penguins.csv').dropna(inplace = False)

input_features = input_file["flipper_length_mm"].values
target         = input_file["body_mass_g"].values

print(input_features)
print(target)

X = input_features.reshape(-1,1)
y_true = target

model = LinearRegression()
model.fit(X, y_true)

plt.show()

example_flipper_length = np.asarray([300, 500])
example_body_mass = model.predict(example_flipper_length.reshape(-1, 1))
print(example_body_mass)

y_pred = model.predict(X)

# Plot the original data and the linear regression line
plt.scatter(X, y_true, color='blue', label='Data Points', marker='.')
plt.plot(X, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('Input')
plt.ylabel('Target')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

# Print out the model parameters (slope and intercept)
print(f"Slope (coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

r_squared = model.score(X, y_true)
print(f"R-squared: {r_squared}")

# Split data into inputs (X) and target (y)
features_to_consider = ["flipper_length_mm" , "bill_depth_mm", "bill_length_mm"]

X = input_file[features_to_consider].values
y_true = input_file["body_mass_g"].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y_true)

# Use the model to make predictions
y_pred = model.predict(X)

# Plot the data and the predicted values (for visualization, we can only plot 2D projections)
# Since we have 2 input features, we can show a projection of the data in a 3D space or a pair of 2D plots.

# Print out the model parameters (coefficients and intercept)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

print(f"R-squared: {model.score(X, y_true)}")

np.random.seed(0)
x = np.linspace(-2, 3, 1000).reshape(-1, 1)
y_true = 2*x**4 - 3*x**3 - 10*x**2 + 0.5*x + 3
y = y_true + 2 * np.random.randn(*y_true.shape)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(x)  # x, x^2, x^3, x^4

# 3. Fit linear regression on polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# 4. Predict
y_pred = model.predict(X_poly)

# 5. Plot results
plt.scatter(x, y, label="Noisy data")
plt.plot(x, y_true, label="True curve", color="green", linewidth=5)
plt.plot(x, y_pred, label="Fitted curve", color="red")
plt.legend()
plt.show()