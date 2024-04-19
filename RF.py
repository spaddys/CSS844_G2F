#### Script to run random forest regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X= pd.read_csv("/mnt/research/css844_2024/G2F/features.csv", low_memory=False, index_col="Pedigree")
y = pd.read_csv("/mnt/research/css844_2024/G2F/labels.csv")


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Define parameters for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best estimator from Grid Search
best_rf_regressor = grid_search.best_estimator_

# Predict on testing data
y_pred = best_rf_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print out best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Plot predicted vs observed values and save as PDF
plt.scatter(X_test, y_test, color='black', label='Observed')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Random Forest Regression')
plt.legend()
plt.savefig('predicted_vs_observed.pdf')