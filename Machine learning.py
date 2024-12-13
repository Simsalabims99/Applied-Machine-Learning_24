import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import multiprocessing

# Utilize maximum CPU power for parallel processing
n_jobs = multiprocessing.cpu_count()

# Load and preprocess data
def load_and_preprocess(file_path, use_cols):
    df = pd.read_csv(file_path, usecols=use_cols, encoding="utf-8")  # Handle Danish letters
    df['Salesprice'] = pd.to_numeric(df['Salesprice'].replace({',': ''}, regex=True), errors='coerce')
    df['Areal (M2)'] = pd.to_numeric(df['Areal (M2)'], errors='coerce')
    df['Rooms'] = pd.to_numeric(df['Rooms'], errors='coerce').astype('Int64')
    df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce').astype('Int64')
    df['Postal code'] = pd.to_numeric(df['Postal code'], errors='coerce').astype('Int64')
    df['Sales year'] = pd.to_datetime(df['Sales date'], format='%d/%m/%Y', errors='coerce').dt.year
    df.drop(columns=['Sales date'], inplace=True)
    df.dropna(inplace=True)
    df['Price per M2'] = df['Salesprice'] / df['Areal (M2)']
    df['Average Price per M2'] = df.groupby(['Postal code', 'Type'])['Price per M2'].transform('mean')
    return df.sample(frac=0.1, random_state=42)

# Target encoding for optimized models
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.encoder = None

    def fit(self, X, y):
        self.encoder = ce.TargetEncoder(cols=['Type'])
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        if self.encoder:
            return self.encoder.transform(X)
        return X

# Cross-validation function
def evaluate_model_with_cv(model, X, y, cv=5):
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=mae_scorer, n_jobs=-1)
    return {'mean_cv_score': -np.mean(cv_scores), 'std_cv_score': np.std(cv_scores)}

# Results display function
def display_results(model_name, metrics):
    print(f"\n{model_name} Results:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"Root Mean Square Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"R²: {metrics['R²']:.2f}")
    print(f"Median Absolute Error (MedAE): {metrics['MedAE']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}")

# Load dataset
file_path = r'C:\Users\7020d\OneDrive\Dokumenter\Privat\CBS\Applied machine learning\Prices slim.csv'
use_cols = ['Areal (M2)', 'Rooms', 'Bathrooms', 'Postal code', 'Salesprice', 'Sales date', 'Type']
df = load_and_preprocess(file_path, use_cols)

# Encode Type column for optimized models
target_encoder = TargetEncoder(target_column='Salesprice')
df = target_encoder.fit_transform(df, df['Salesprice'])

# Define features for baseline models (use one-hot encoding for Type)
baseline_features = ['Areal (M2)', 'Rooms', 'Bathrooms', 'Postal code', 'Sales year']
X_baseline = pd.get_dummies(df[baseline_features + ['Type']], drop_first=True)
y = df['Salesprice']
X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)

# Define features for optimized models (use target encoding for Type)
optimized_features = ['Areal (M2)', 'Rooms', 'Bathrooms', 'Postal code', 'Sales year', 'Average Price per M2', 'Type']
X_optimized = df[optimized_features]
scaler = RobustScaler()
X_optimized_scaled = scaler.fit_transform(X_optimized)
X_train_optimized, X_test_optimized, y_train, y_test = train_test_split(X_optimized_scaled, y, test_size=0.2, random_state=42)

# Baseline models
baseline_models = {
    'Linear Regression (Baseline)': LinearRegression(),
    'Random Forest (Baseline)': RandomForestRegressor(random_state=42, n_jobs=n_jobs),
    'Gradient Boosting (Baseline)': GradientBoostingRegressor(random_state=42),
}

# Optimized models with hyperparameter tuning
param_distributions = {
    'Random Forest': {
        'n_estimators': [25, 50, 100],
        'max_depth': [3, 6, 9],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
    },
    'Ridge Regression': {
        'alpha': np.logspace(-3, 3, 10),
    },
}

optimized_models = {}
for model_name, params in param_distributions.items():
    if model_name == 'Ridge Regression':
        model = Ridge()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
    else:
        model = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        model, param_distributions=params, n_iter=10, cv=5, scoring='neg_mean_absolute_error', random_state=42, n_jobs=n_jobs
    )
    random_search.fit(X_train_optimized, y_train)
    optimized_models[model_name] = random_search.best_estimator_

# Combine baseline and optimized models
all_models = {**baseline_models, **optimized_models}

# Evaluate all models
results = {}
for model_name, model in all_models.items():
    X_train = X_train_baseline if 'Baseline' in model_name else X_train_optimized
    X_test = X_test_baseline if 'Baseline' in model_name else X_test_optimized
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred),
        'MedAE': median_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        'y_pred': y_pred,
    }
    display_results(model_name, results[model_name])


# Define the mapping of display names to actual keys in the results dictionary
plot_order_mapping = {
    'Linear Regression (Baseline)': 'Linear Regression (Baseline)',  # Actual key in results
    'Random Forest (Baseline)': 'Random Forest (Baseline)',          # Actual key in results
    'Gradient Boosting (Baseline)': 'Gradient Boosting (Baseline)',  # Actual key in results
    'Ridge Regression': 'Ridge Regression',                         # Actual key in results
    'Random Forest (Optimized)': 'Random Forest',                   # Actual key in results
    'Gradient Boosting (Optimized)': 'Gradient Boosting',           # Actual key in results
}

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Comparisons: Actual vs Predicted', fontsize=16, fontweight='bold')
axes = axes.ravel()

# Define custom ticks for the x-axis
custom_ticks = np.arange(0, 16e6, 3e6)  # Ticks at 0, 3M, 6M, 9M, 12M, and 15M

# Plot in the specified order
for ax, (display_name, actual_key) in zip(axes, plot_order_mapping.items()):
    if actual_key not in results:
        print(f"Model {actual_key} is not in results.")
        continue
    metrics = results[actual_key]
    y_pred = metrics['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual', color='blue')
    ax.plot([0, 15e6], [0, 15e6], 'r--', label='Perfect Prediction')  # Diagonal line
    ax.set_xlim(0, 15e6)  # Set x-axis limits
    ax.set_ylim(0, 15e6)  # Set y-axis limits
    ax.set_title(display_name, fontsize=12)  # Use display name here
    ax.set_xlabel('Actual Sales Price (in mio.)', fontsize=10)
    ax.set_ylabel('Predicted Sales Price (in mio.)', fontsize=10)
    
    # Set custom ticks and format them as "0 mio.", "3 mio.", etc.
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels([f'{int(x/1e6)} mio.' for x in custom_ticks])
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels([f'{int(y/1e6)} mio.' for y in custom_ticks])
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
