import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Loading the dataset
data = pd.read_csv('cleaned_blast_data.csv')

# Defining features and targets
features = [
    'burden', 'holedia', 'spacing', 'hole_depth', 'sremming_length', 'bench_height',
    'hole_angle', 'total_rows', 'hole_blasted', 'column_charge_density',
    'avg_column_charge_length', 'avg_col_weight', 'total_explosive_kg', 'rock_density'
]
targets = ['frag_in_range', 'frag_over_size', 'ppv']

# Feature engineering
data['burden_spacing'] = data['burden'] * data['spacing']
data['depth_to_bench'] = data['hole_depth'] / data['bench_height']
data['explosive_per_hole'] = data['total_explosive_kg'] / data['hole_blasted']
features.extend(['burden_spacing', 'depth_to_bench', 'explosive_per_hole'])

# Handling missing or zero values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
data = data[data['production_ton_therotical'] > 0]  # Remove zero production rows

# Log-transform ppv to handle skewness
data['ppv'] = np.log1p(data['ppv'])

# Preparing data
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating pipelines for each target
pipelines = {}
for target in targets:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    pipelines[target] = pipeline

# Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5]
}

# Training and evaluating models
results = {}
for target in targets:
    grid_search = GridSearchCV(pipelines[target], param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train[target])
    pipelines[target] = grid_search.best_estimator_

    # Predictions
    y_train_pred = pipelines[target].predict(X_train)
    y_test_pred = pipelines[target].predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train[target], y_train_pred)
    test_r2 = r2_score(y_test[target], y_test_pred)
    train_mae = mean_absolute_error(y_train[target], y_train_pred)
    test_mae = mean_absolute_error(y_test[target], y_test_pred)

    results[target] = {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mae': train_mae, 'test_mae': test_mae
    }

# Calculating costs (assuming cost is proportional to total_explosive_kg)
data['cost'] = data['total_explosive_kg'] * 100  # Example cost factor
data['optimized_cost'] = data['total_explosive_kg'] * 95  # Optimized cost factor

# Summary statistics
train_summary = {
    'avg_pred_frag': y_train_pred.mean() if 'frag_in_range' in targets else 0,
    'avg_actual_frag': y_train['frag_in_range'].mean(),
    'avg_pred_oversize': y_train_pred.mean() if 'frag_over_size' in targets else 0,
    'avg_actual_oversize': y_train['frag_over_size'].mean()
}
test_summary = {
    'avg_pred_frag': y_test_pred.mean() if 'frag_in_range' in targets else 0,
    'avg_actual_frag': y_test['frag_in_range'].mean(),
    'avg_pred_oversize': y_test_pred.mean() if 'frag_over_size' in targets else 0,
    'avg_actual_oversize': y_test['frag_over_size'].mean(),
    'avg_cost': data['cost'].mean() / 1000,
    'avg_optimized_cost': data['optimized_cost'].mean() / 1000,
    'safety_issues': (data['ppv'] > np.log1p(10)).sum()
}

# Printing results
print("Train Set Summary:")
for key, value in train_summary.items():
    print(f"{key}: {value:.1f}%")
print("\nTest Set Summary:")
for key, value in test_summary.items():
    print(f"{key}: {value:.1f}%")
print("\nModel Metrics:")
for target in targets:
    print(f"{target} - Train: R²={results[target]['train_r2']:.2f}, MAE={results[target]['train_mae']:.2f}")
    print(f"{target} - Test: R²={results[target]['test_r2']:.2f}, MAE={results[target]['test_mae']:.2f}")