import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from itertools import combinations
import multiprocessing

# Utilize maximum CPU power for parallel processing
n_jobs = multiprocessing.cpu_count()

# Load data and preprocess
file_path = r'C:\Users\7020d\OneDrive\Dokumenter\Privat\CBS\Applied machine learning\Prices slim.csv'
use_cols = ['Areal (M2)', 'Rooms', 'Bathrooms', 'Postal code', 'Salesprice', 'Sales date', 'Type']
df = pd.read_csv(file_path, usecols=use_cols)

# Clean and convert columns to numeric types
df['Salesprice'] = pd.to_numeric(df['Salesprice'].replace({',': ''}, regex=True), errors='coerce')
df['Areal (M2)'] = pd.to_numeric(df['Areal (M2)'], errors='coerce')
df['Rooms'] = pd.to_numeric(df['Rooms'], errors='coerce').astype('int16')
df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce').astype('int16')
df['Postal code'] = pd.to_numeric(df['Postal code'], errors='coerce').astype('int32')

# Extract year from 'Sales date' and add it as a new feature
df['Sales year'] = pd.to_datetime(df['Sales date'], format='%d/%m/%Y', errors='coerce').dt.year
df.drop(columns=['Sales date'], inplace=True)  # Drop original date column
df.dropna(inplace=True)  # Remove rows with any missing values

# Calculate average Price per M2 by Postal code and Type
df['Price per M2'] = df['Salesprice'] / df['Areal (M2)']
average_price_per_m2 = df.groupby(['Postal code', 'Type'])['Price per M2'].transform('mean')
df['Average Price per M2'] = average_price_per_m2

# Reduce dataset size for faster execution (sampling 10%)
df = df.sample(frac=0.1, random_state=42)

# Define all possible features
all_features = ['Areal (M2)', 'Rooms', 'Bathrooms', 'Postal code', 'Sales year', 'Average Price per M2']
y = df['Salesprice']

# Dictionary to store best results for each combination
best_results = {}

# Loop through all combinations of features to find the best performing model
for i in range(1, len(all_features) + 1):
    for combo in combinations(all_features, i):
        # Define features for the current combination
        X = df[list(combo)]

        # Apply RobustScaler to features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize Random Forest model
        random_forest = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
        random_forest.fit(X_train, y_train)

        # Make predictions
        y_pred = random_forest.predict(X_test)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)

        # Store the results
        best_results[combo] = mae

# Find the best feature combination based on MAE
best_features = min(best_results, key=best_results.get)
print(f"Best Feature Combination: {best_features}")
print(f"MAE: {best_results[best_features]:.2f}")

# Use the best feature combination for final model training
X = df[list(best_features)]
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest using RandomizedSearchCV
param_distributions_rf = {
    'n_estimators': [25, 50, 75, 100, 150],
    'max_depth': [3, 6, 9, 12, 15],
    'min_samples_split': [2, 4, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
random_forest = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
rf_random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_distributions_rf, n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=n_jobs)
rf_random_search.fit(X_train, y_train)

# Best Random Forest model
best_rf = rf_random_search.best_estimator_

# Hyperparameter tuning for Gradient Boosting using RandomizedSearchCV
param_distributions_gb = {
    'n_estimators': [50, 75, 100, 150, 200],
    'max_depth': [3, 5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.075, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}
gb = GradientBoostingRegressor(random_state=42)
gb_random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_distributions_gb, n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=n_jobs)
gb_random_search.fit(X_train, y_train)

# Best Gradient Boosting model
best_gb = gb_random_search.best_estimator_

# Feature Importance using RandomForestClassifier to identify key features
rf_classifier = RandomForestClassifier(random_state=42, max_leaf_nodes=30, n_jobs=n_jobs)
rf_classifier.fit(X, y > y.median())  # Use median sales price as a threshold for classification
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
print("\nFeature Importances from RandomForestClassifier:")
print(feature_importances.sort_values(ascending=False))

# Initialize models
models = {
    'Linear Regression': LinearRegression(n_jobs=n_jobs),
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb
}

# Dictionary to store results
results = {}

# Train, predict, and calculate metrics for each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Retrieve model details
    feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    intercept = model.intercept_ if hasattr(model, 'intercept_') else None
    
    # Store the results
    results[model_name] = {
        'MAE': mae,
        'Feature Importances or Coefficients': feature_importance,
        'Intercept': intercept,
        'y_pred': y_pred  # Store predictions
    }

# Find the best model based on MAE
best_model_name = min(results, key=lambda x: results[x]['MAE'])

# Display the results
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"Feature Importances or Coefficients: {metrics['Feature Importances or Coefficients']}")
    print(f"Intercept: {metrics['Intercept']}")

# Plot the predictions vs actual values for each model in a grid
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Model Comparison - Actual vs Predicted Sales Price', fontsize=16, fontweight='bold')

for ax, (model_name, metrics) in zip(axes, results.items()):
    y_pred = metrics['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolor='k', s=20, label='Predicted vs Actual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Ideal Fit (y = x)')
    ax.set_xlabel('Actual Sales Price (in millions)', fontsize=10)
    ax.set_xticks(np.arange(500_000, 20_000_001, step=5_000_000))
    ax.set_xticklabels([f'{i//1_000_000}M' for i in np.arange(500_000, 20_000_001, step=5_000_000)], fontsize=8)
    ax.set_xlim(500_000, 20_000_000)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

axes[0].set_ylabel('Predicted Sales Price (in millions)', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
