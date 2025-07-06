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
data = pd.read_csv('cleaned_blast_data.csv')

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

# Clean data
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
data = data[data['production_ton_therotical'] > 0]
data['ppv'] = np.log1p(data['ppv'])  # log transform

# Add cost metrics
data['cost'] = data['total_explosive_kg'] * 100
data['optimized_cost'] = data['total_explosive_kg'] * 95
data['cost_savings'] = data['cost'] - data['optimized_cost']

# Prepare X and y
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer([('num', StandardScaler(), features)])

# Training pipeline and evaluation
results = {}
pipelines = {}
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, None]
}

for target in targets:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train[target])
    best_model = grid.best_estimator_
    pipelines[target] = best_model

    # Save model
    joblib.dump(best_model, f'rf_model_{target}.joblib')

    y_pred = best_model.predict(X_test)
    results[target] = {
        'test_r2': r2_score(y_test[target], y_pred),
        'test_mae': mean_absolute_error(y_test[target], y_pred),
        'y_test': y_test[target],
        'y_pred': y_pred
    }

# Safety & summaries
data['unsafe_blast'] = data['ppv'] > np.log1p(10)
unsafe_count = data['unsafe_blast'].sum()

summary = {
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

# Plot results
plt.figure(figsize=(15, 10))
for i, target in enumerate(targets, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=results[target]['y_test'], y=results[target]['y_pred'], alpha=0.6)
    plt.plot([results[target]['y_test'].min(), results[target]['y_test'].max()],
             [results[target]['y_test'].min(), results[target]['y_test'].max()], 'r--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'{target}: R²={results[target]["test_r2"]:.2f}')

plt.subplot(2, 2, 4)
feat_imp = pd.Series(
    pipelines['frag_in_range'].named_steps['rf'].feature_importances_, index=features
).sort_values(ascending=False)[:8]
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top Feature Importance")
plt.tight_layout()
plt.savefig('blast_model_plots.png')
plt.close()

# Print results
print("\nSummary:")
for k, v in summary.items():
    print(f"{k}: {v:.2f}" + ('%' if 'frag' in k or 'oversize' in k else ''))
for target in targets:
    print(f"{target} -> R²: {results[target]['test_r2']:.2f}, MAE: {results[target]['test_mae']:.2f}")
print(f"\n{unsafe_count} unsafe blasts detected.")
print("\nModels saved as: rf_model_frag_in_range.joblib, rf_model_frag_over_size.joblib, rf_model_ppv.joblib")
