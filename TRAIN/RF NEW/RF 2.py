import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load dataset
try:
    data = pd.read_csv('cleaned_blast_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Ensure 'cleaned_blast_data.csv' is in the working directory.")

# Define features and targets
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

# Handle missing or invalid values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
data = data[data['production_ton_therotical'] > 0]

# Log-transform ppv
data['ppv'] = np.log1p(data['ppv'])

# Cost calculations
data['cost'] = data['total_explosive_kg'] * 100
data['optimized_cost'] = data['total_explosive_kg'] * 95
data['cost_savings'] = data['cost'] - data['optimized_cost']

# Prepare data
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipelines
pipelines = {}
preprocessor = ColumnTransformer([('num', StandardScaler(), features)])
for target in targets:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    pipelines[target] = pipeline

# Simplified hyperparameter tuning
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, None]
}

# Train and evaluate
results = {}
for target in targets:
    grid_search = GridSearchCV(pipelines[target], param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train[target])
    pipelines[target] = grid_search.best_estimator_

    # Predictions
    y_test_pred = pipelines[target].predict(X_test)

    # Metrics
    test_r2 = r2_score(y_test[target], y_test_pred)
    test_mae = mean_absolute_error(y_test[target], y_test_pred)

    results[target] = {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'y_test': y_test[target],
        'y_pred': y_test_pred
    }

    # Save model
    joblib.dump(pipelines[target], f'rf_model_{target}.joblib')

# Safety analysis
safety_threshold = np.log1p(10)
data['unsafe_blast'] = data['ppv'] > safety_threshold
unsafe_count = data['unsafe_blast'].sum()

# Summaries
test_summary = {
    'avg_pred_frag': results['frag_in_range']['y_pred'].mean(),
    'avg_actual_frag': y_test['frag_in_range'].mean(),
    'avg_pred_oversize': results['frag_over_size']['y_pred'].mean(),
    'avg_actual_oversize': y_test['frag_over_size'].mean(),
    'avg_pred_ppv': np.expm1(results['ppv']['y_pred']).mean(),
    'avg_actual_ppv': np.expm1(y_test['ppv']).mean(),
    'avg_cost': data['cost'].mean() / 1000,
    'cost_savings': data['cost_savings'].mean() / 1000,
    'unsafe_blasts': unsafe_count
}

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Actual vs Predicted
for i, target in enumerate(targets, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=results[target]['y_test'], y=results[target]['y_pred'], alpha=0.6)
    plt.plot([results[target]['y_test'].min(), results[target]['y_test'].max()],
             [results[target]['y_test'].min(), results[target]['y_test'].max()], 'r--')
    plt.xlabel(f'Actual {target}' + (' (log)' if target == 'ppv' else ' (%)'))
    plt.ylabel(f'Predicted {target}' + (' (log)' if target == 'ppv' else ' (%)'))
    plt.title(f'{target}: R²={results[target]["test_r2"]:.2f}')
    plt.text(0.05, 0.95, f'R²={results[target]["test_r2"]:.2f}', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# 2. Feature Importance
plt.subplot(2, 2, 4)
feature_importance = pd.Series(
    pipelines['frag_in_range'].named_steps['rf'].feature_importances_, index=features
).sort_values(ascending=False)[:8]
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title(f'Feature Importance (R²={results["frag_in_range"]["test_r2"]:.2f})')
plt.xlabel('Importance')

plt.tight_layout()
plt.savefig('blast_model_plots.png')
plt.close()

# Print summary
print("Test Set Summary:")
for key, value in test_summary.items():
    print(f"{key}: {value:.1f}" + ('%' if 'frag' in key or 'oversize' in key else ''))
print("\nModel Metrics:")
for target in targets:
    print(f"{target} - Test: R²={results[target]['test_r2']:.2f}, MAE={results[target]['test_mae']:.2f}")
print(f"\nSafety: {unsafe_count} unsafe blasts")
print(f"Cost Savings: {test_summary['cost_savings']:.1f}k per blast")

print("\nModels saved as 'rf_model_<target>.joblib'")
print("Plots saved as 'blast_model_plots.png'")