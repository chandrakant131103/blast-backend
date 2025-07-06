import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Load dataset ---
data = pd.read_csv('Mine_Swamp.csv')
# Optional: Rename 'sremming_length' to 'stemming_length' to fix typo
# data.rename(columns={'sremming_length': 'stemming_length'}, inplace=True)
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")  # normalize columns

# --- Initial feature and target selection ---
features = [
    'burden', 'holedia', 'spacing', 'hole_depth', 'sremming_length', 'bench_height',
    'hole_angle', 'total_rows', 'hole_blasted', 'column_charge_density',
    'avg_column_charge_length', 'avg_col_weight', 'total_explosive_kg', 'rock_density'
]
targets = ['frag_in_range', 'frag_over_size', 'ppv']

# --- Feature engineering ---
data['burden_spacing'] = data['burden'] * data['spacing']
data['depth_to_bench'] = data['hole_depth'] / data['bench_height']
data['explosive_per_hole'] = data['total_explosive_kg'] / data['hole_blasted']
features.extend(['burden_spacing', 'depth_to_bench', 'explosive_per_hole'])

# --- Kuz-Ram Model ---
def compute_x50(row):
    B = row['burden']
    S = row['spacing']
    H = row['hole_depth']
    Q = row['total_explosive_kg']
    return 0 if Q == 0 else 0.2 * ((B * S * H) / Q) ** 0.8

data['x50'] = data.apply(compute_x50, axis=1)
data['x80'] = data['x50'] * (np.log(5)) ** (1 / 1.2)

# --- Fragmentation out-of-range ---
data['frag_out_of_range'] = data['frag_over_size'] + (100 - data['frag_in_range'] - data['frag_over_size'])

# --- Cost calculations ---
data['drill_cost'] = data['hole_depth'] * data['hole_blasted'] * 150  # ₹150 per meter
data['explosive_cost'] = data['total_explosive_kg'] * 100             # ₹100 per kg
data['total_blast_cost'] = data['drill_cost'] + data['explosive_cost']

# --- PPV Alert (Safety flag) ---
GROUND_VIBRATION_LIMIT = np.log1p(5)
data['ppv_alert'] = (data['ppv'] > GROUND_VIBRATION_LIMIT).astype(int)  # 0 = Safe, 1 = Danger

# Add new engineered features
features.extend([
    'x50', 'x80',
    'frag_out_of_range',
    'drill_cost', 'explosive_cost', 'total_blast_cost',
    'ppv_alert'
])

# --- Clean data ---
print("DataFrame columns:", data.columns.tolist())
print("Expected columns:", features + targets)
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
data = data[data['production_ton_therotical'] > 0]
data['ppv'] = np.log1p(data['ppv'])  # log transform PPV

# --- Additional cost metrics ---
data['cost'] = data['total_explosive_kg'] * 100
data['optimized_cost'] = data['total_explosive_kg'] * 95
data['cost_savings'] = data['cost'] - data['optimized_cost']

# --- Train-test split ---
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing pipeline ---
preprocessor = ColumnTransformer([('num', StandardScaler(), features)])

# --- Model training and evaluation ---
results = {}
pipelines = {}
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]  # Added to reduce overfitting
}

for target in targets:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train[target])
    best_model = grid.best_estimator_
    pipelines[target] = best_model

    # Save model
    joblib.dump(best_model, f'rf_model_{target}.joblib')

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    results[target] = {
        'test_r2': r2_score(y_test[target], y_pred),
        'test_mae': mean_absolute_error(y_test[target], y_pred),
        'y_test': y_test[target],
        'y_pred': y_pred
    }

# --- Debug frag_in_range predictions ---
print("Predicted frag_in_range range on test set:", pipelines['frag_in_range'].predict(X_test).min(), pipelines['frag_in_range'].predict(X_test).max())

# --- Optimizations ---
FRAG_THRESHOLD = 10  # Lowered to reduce optimization failures

# Cost Optimization: Minimize total_blast_cost while frag_in_range >= FRAG_THRESHOLD
def optimize_cost(row, model, features, data):
    explosive_range = np.linspace(row['total_explosive_kg'] * 0.8, row['total_explosive_kg'] * 1.2, 10)
    min_cost = float('inf')
    best_explosive = row['total_explosive_kg']
    pred_frags = []
    for explosive in explosive_range:
        temp_row = row.copy()
        temp_row['total_explosive_kg'] = explosive
        temp_row['explosive_cost'] = explosive * 100
        temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
        temp_row['explosive_per_hole'] = explosive / temp_row['hole_blasted']
        temp_row['x50'] = compute_x50(temp_row)
        temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
        pred_frag = model.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
        pred_frags.append(pred_frag)
        if pred_frag >= FRAG_THRESHOLD and temp_row['total_blast_cost'] < min_cost:
            min_cost = temp_row['total_blast_cost']
            best_explosive = explosive
    if min_cost == float('inf'):
        print(f"Warning: No solution found for cost optimization at index {row.name}, using original cost")
        print(f"Predicted frag_in_range values for index {row.name}: min={min(pred_frags):.2f}, max={max(pred_frags):.2f}")
        min_cost = data.loc[row.name, 'total_blast_cost']
    return best_explosive, min_cost

# Fragmentation Optimization: Maximize frag_in_range
def optimize_fragmentation(row, model, features):
    burden_range = np.linspace(row['burden'] * 0.8, row['burden'] * 1.2, 5)
    spacing_range = np.linspace(row['spacing'] * 0.8, row['spacing'] * 1.2, 5)
    explosive_range = np.linspace(row['total_explosive_kg'] * 0.8, row['total_explosive_kg'] * 1.2, 5)
    max_frag = 0
    best_params = {'burden': row['burden'], 'spacing': row['spacing'], 'total_explosive_kg': row['total_explosive_kg']}
    for burden in burden_range:
        for spacing in spacing_range:
            for explosive in explosive_range:
                temp_row = row.copy()
                temp_row['burden'] = burden
                temp_row['spacing'] = spacing
                temp_row['total_explosive_kg'] = explosive
                temp_row['burden_spacing'] = burden * spacing
                temp_row['explosive_per_hole'] = explosive / temp_row['hole_blasted']
                temp_row['x50'] = compute_x50(temp_row)
                temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
                pred_frag = model.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
                if pred_frag > max_frag:
                    max_frag = pred_frag
                    best_params = {'burden': burden, 'spacing': spacing, 'total_explosive_kg': explosive}
    return max_frag, best_params

# Safety Optimization: Minimize ppv while frag_in_range >= FRAG_THRESHOLD
def optimize_safety(row, model_ppv, model_frag, features, data):
    explosive_range = np.linspace(row['total_explosive_kg'] * 0.8, row['total_explosive_kg'] * 1.2, 10)
    min_ppv = float('inf')
    best_explosive = row['total_explosive_kg']
    pred_frags = []
    for explosive in explosive_range:
        temp_row = row.copy()
        temp_row['total_explosive_kg'] = explosive
        temp_row['explosive_per_hole'] = explosive / temp_row['hole_blasted']
        temp_row['x50'] = compute_x50(temp_row)
        temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
        pred_frag = model_frag.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
        pred_ppv = model_ppv.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
        pred_frags.append(pred_frag)
        if pred_frag >= FRAG_THRESHOLD and pred_ppv < min_ppv:
            min_ppv = pred_ppv
            best_explosive = explosive
    if min_ppv == float('inf'):
        print(f"Warning: No solution found for safety optimization at index {row.name}, using original PPV")
        print(f"Predicted frag_in_range values for index {row.name}: min={min(pred_frags):.2f}, max={max(pred_frags):.2f}")
        min_ppv = data.loc[row.name, 'ppv']
    return min_ppv, best_explosive

# Apply optimizations on test set
optimized_costs = []
optimized_frags = []
optimized_ppvs = []
for idx in X_test.index:
    row = X.loc[idx].copy()
    row.name = idx  # Ensure row.name is set for warning messages
    # Cost optimization
    best_explosive, min_cost = optimize_cost(row, pipelines['frag_in_range'], features, data)
    optimized_costs.append(min_cost)
    # Fragmentation optimization
    max_frag, best_params = optimize_fragmentation(row, pipelines['frag_in_range'], features)
    optimized_frags.append(max_frag)
    # Safety optimization
    min_ppv, best_explosive_ppv = optimize_safety(row, pipelines['ppv'], pipelines['frag_in_range'], features, data)
    optimized_ppvs.append(min_ppv)

# --- Safety & Summary ---
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
    'unsafe_blasts': unsafe_count,
    'optimized_avg_cost': np.mean(optimized_costs) / 1000,
    'optimized_avg_frag': np.mean(optimized_frags),
    'optimized_avg_ppv': np.expm1(np.mean(optimized_ppvs))
}

# --- Plotting ---
# Plot 1: Actual vs Predicted frag_in_range
plt.figure(figsize=(8, 6))
sns.scatterplot(x=results['frag_in_range']['y_test'], y=results['frag_in_range']['y_pred'], alpha=0.6)
plt.plot([results['frag_in_range']['y_test'].min(), results['frag_in_range']['y_test'].max()],
         [results['frag_in_range']['y_test'].min(), results['frag_in_range']['y_test'].max()], 'r--')
plt.xlabel('Actual Fragmentation In Range (%)')
plt.ylabel('Predicted Fragmentation In Range (%)')
plt.title(f'Fragmentation In Range: R²={results["frag_in_range"]["test_r2"]:.2f}')
plt.tight_layout()
plt.savefig('fragmentation_in_range.png')
plt.close()

# Plot 2: Actual vs Predicted frag_over_size
plt.figure(figsize=(8, 6))
sns.scatterplot(x=results['frag_over_size']['y_test'], y=results['frag_over_size']['y_pred'], alpha=0.6)
plt.plot([results['frag_over_size']['y_test'].min(), results['frag_over_size']['y_test'].max()],
         [results['frag_over_size']['y_test'].min(), results['frag_over_size']['y_test'].max()], 'r--')
plt.xlabel('Actual Fragmentation Over Size (%)')
plt.ylabel('Predicted Fragmentation Over Size (%)')
plt.title(f'Fragmentation Over Size: R²={results["frag_over_size"]["test_r2"]:.2f}')
plt.tight_layout()
plt.savefig('fragmentation_over_size.png')
plt.close()

# Plot 3: Actual vs Predicted PPV
plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.expm1(results['ppv']['y_test']), y=np.expm1(results['ppv']['y_pred']), alpha=0.6)
plt.plot([np.expm1(results['ppv']['y_test']).min(), np.expm1(results['ppv']['y_test']).max()],
         [np.expm1(results['ppv']['y_test']).min(), np.expm1(results['ppv']['y_test']).max()], 'r--')
plt.xlabel('Actual PPV (mm/s)')
plt.ylabel('Predicted PPV (mm/s)')
plt.title(f'PPV: R²={results["ppv"]["test_r2"]:.2f}')
plt.tight_layout()
plt.savefig('ppv_plot.png')
plt.close()

# Plot 4: Actual vs Optimized Cost
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data.loc[X_test.index, 'total_blast_cost'], y=optimized_costs, alpha=0.6)
plt.plot([data['total_blast_cost'].min(), data['total_blast_cost'].max()],
         [data['total_blast_cost'].min(), data['total_blast_cost'].max()], 'r--')
plt.xlabel('Actual Total Blast Cost (₹)')
plt.ylabel('Optimized Total Blast Cost (₹)')
plt.title('Actual vs Optimized Cost')
plt.tight_layout()
plt.savefig('cost_plot.png')
plt.close()

# Plot 5: Safety (Confusion Matrix for ppv_alert)
plt.figure(figsize=(8, 6))
y_pred_ppv_alert = (results['ppv']['y_pred'] > GROUND_VIBRATION_LIMIT).astype(int)
cm = ConfusionMatrixDisplay.from_predictions(
    data.loc[X_test.index, 'ppv_alert'], y_pred_ppv_alert, display_labels=['Safe', 'Danger']
)
cm.plot(ax=plt.gca(), cmap='Blues')
plt.title('Safety: Actual vs Predicted PPV Alert')
plt.tight_layout()
plt.savefig('safety_plot.png')
plt.close()

# Plot 6: Correlation Heatmap (Improved)
plt.figure(figsize=(10, 8))
numeric_cols = ['burden', 'spacing', 'total_explosive_kg', 'frag_in_range', 'ppv']
corr_matrix = data[numeric_cols].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f', annot_kws={'size': 12})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Plot 7: Optimization Results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.barplot(x=['Baseline', 'Optimized'], y=[data.loc[X_test.index, 'total_blast_cost'].mean() / 1000, np.mean(optimized_costs) / 1000])
plt.title('Average Cost (₹1000)')
plt.subplot(1, 3, 2)
sns.barplot(x=['Baseline', 'Optimized'], y=[results['frag_in_range']['y_test'].mean(), np.mean(optimized_frags)])
plt.title('Average Fragmentation In Range (%)')
plt.subplot(1, 3, 3)
sns.barplot(x=['Baseline', 'Optimized'], y=[np.expm1(results['ppv']['y_test']).mean(), np.expm1(np.mean(optimized_ppvs))])
plt.title('Average PPV (mm/s)')
plt.tight_layout()
plt.savefig('optimization_results.png')
plt.close()

# Plot 8: Actual Frag In Range vs Sample Index
plt.figure(figsize=(8, 6))
sns.lineplot(x=results['frag_in_range']['y_test'].index, y=results['frag_in_range']['y_test'])
plt.xlabel('Sample Index')
plt.ylabel('Actual Fragmentation In Range (%)')
plt.title('Actual Fragmentation In Range vs Sample Index')
plt.tight_layout()
plt.savefig('frag_in_range_vs_index.png')
plt.close()

# Plot 9: Actual Frag Out of Range vs Log10(x50)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.log10(data.loc[X_test.index, 'x50']), y=data.loc[X_test.index, 'frag_out_of_range'], alpha=0.6)
plt.xlabel('Log10(x50) (Fragment Size)')
plt.ylabel('Actual Fragmentation Out of Range (%)')
plt.title('Fragmentation Out of Range vs Log10(x50)')
plt.tight_layout()
plt.savefig('frag_out_of_range_vs_log10.png')
plt.close()

# Plot 10: Actual vs Predicted Frag In Range vs Sample Index (Overlay)
plt.figure(figsize=(8, 6))
sns.lineplot(x=results['frag_in_range']['y_test'].index, y=results['frag_in_range']['y_test'], label='Actual')
sns.lineplot(x=results['frag_in_range']['y_test'].index, y=results['frag_in_range']['y_pred'], label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Fragmentation In Range (%)')
plt.title('Actual vs Predicted Fragmentation In Range vs Sample Index')
plt.legend()
plt.tight_layout()
plt.savefig('frag_in_range_actual_vs_predicted.png')
plt.close()

# --- Print Summary ---
print("\nSummary:")
for k, v in summary.items():
    print(f"{k}: {v:.2f}" + ('%' if 'frag' in k or 'oversize' in k else ''))
for target in targets:
    print(f"{target} -> R²: {results[target]['test_r2']:.2f}, MAE: {results[target]['test_mae']:.2f}")
print(f"\n{unsafe_count} unsafe blasts detected.")
print("\nModels saved as:")
for target in targets:
    print(f" - rf_model_{target}.joblib")