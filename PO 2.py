import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load and preprocess dataset
data = pd.read_csv('Mine_Swamp.csv')
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
data.rename(columns={
    'sremming_length': 'stemming_length',
    'holedia': 'hole_diameter',
    'hole_blasted': 'holes_blasted',
    'avg_col_weight': 'avg_column_weight'
}, inplace=True)

# Feature engineering
features = [
    'burden', 'hole_diameter', 'spacing', 'hole_depth', 'stemming_length', 'bench_height',
    'hole_angle', 'total_rows', 'holes_blasted', 'column_charge_density',
    'avg_column_charge_length', 'avg_column_weight', 'total_explosive_kg', 'rock_density',
    'burden_spacing', 'depth_to_bench', 'explosive_per_hole', 'x50', 'x80',
    'frag_out_of_range', 'drill_cost', 'explosive_cost', 'total_blast_cost', 'ppv_alert'
]
targets = ['frag_in_range', 'frag_over_size', 'ppv']

data['burden_spacing'] = data['burden'] * data['spacing']
data['depth_to_bench'] = data['hole_depth'] / data['bench_height']
data['explosive_per_hole'] = data['total_explosive_kg'] / data['holes_blasted']

def compute_x50(row):
    B = row['burden']
    S = row['spacing']
    H = row['hole_depth']
    Q = row['total_explosive_kg']
    return 0 if Q == 0 else 0.2 * ((B * S * H) / (Q + 1e-10)) ** 0.8

data['x50'] = data.apply(compute_x50, axis=1)
data['x80'] = data['x50'] * (np.log(5)) ** (1 / 1.2)
data['frag_out_of_range'] = data['frag_over_size'] + (100 - data['frag_in_range'] - data['frag_over_size'])
data['drill_cost'] = data['hole_depth'] * data['holes_blasted'] * 150
data['explosive_cost'] = data['total_explosive_kg'] * 100
data['total_blast_cost'] = data['drill_cost'] + data['explosive_cost']
GROUND_VIBRATION_LIMIT = np.log1p(5)
data['ppv_alert'] = (data['ppv'] > GROUND_VIBRATION_LIMIT).astype(int)
data['cost'] = data['total_explosive_kg'] * 100
data['optimized_cost'] = data['total_explosive_kg'] * 95
data['cost_savings'] = data['cost'] - data['optimized_cost']

# Clean data
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
data = data[data['production_ton_therotical'] > 0]
data['ppv'] = np.log1p(data['ppv'])

# Log zero hole_angle occurrences
print(f"Number of rows with hole_angle == 0: {len(data[data['hole_angle'] == 0])}")

# Validate data at index 296
if 296 in data.index:
    print(f"Data at index 296:\n{data.loc[296]}")
else:
    print("Index 296 not found in dataset")

# Train-test split
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test indices for backend
joblib.dump(X_test.index, 'X_test_indices.joblib')

# Preprocessing pipeline
preprocessor = ColumnTransformer([('num', StandardScaler(), features)])
joblib.dump(preprocessor, 'preprocessor.joblib')

# Train models
pipelines = {}
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}
results = {}
for target in targets:
    pipeline = Pipeline([('preprocessor', preprocessor), ('rf', RandomForestRegressor(random_state=42))])
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train[target])
    best_model = grid.best_estimator_
    pipelines[target] = best_model
    joblib.dump(best_model, f'rf_model_{target}.joblib')
    y_pred = best_model.predict(X_test)
    results[target] = {
        'test_r2': r2_score(y_test[target], y_pred),
        'test_mae': mean_absolute_error(y_test[target], y_pred),
        'y_test': y_test[target].values,
        'y_pred': y_pred
    }

# Optimization functions
FRAG_THRESHOLD = 5

def optimize_cost(row, model, features, data):
    explosive_range = np.linspace(row['total_explosive_kg'] * 0.5, row['total_explosive_kg'] * 1.5, 20)
    holes_range = np.linspace(max(1, row['holes_blasted'] * 0.5), row['holes_blasted'] * 1.5, 10, dtype=int)
    min_cost = float('inf')
    best_explosive = row['total_explosive_kg']
    best_holes = row['holes_blasted']
    pred_frags = []
    for explosive in explosive_range:
        for holes in holes_range:
            temp_row = row.copy()
            temp_row['total_explosive_kg'] = explosive
            temp_row['holes_blasted'] = holes
            temp_row['explosive_cost'] = explosive * 100
            temp_row['drill_cost'] = temp_row['hole_depth'] * holes * 150
            temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
            temp_row['explosive_per_hole'] = explosive / holes if holes > 0 else 0
            temp_row['x50'] = compute_x50(temp_row)
            temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
            pred_frag = model.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
            pred_frags.append(pred_frag)
            if pred_frag >= FRAG_THRESHOLD and temp_row['total_blast_cost'] < min_cost:
                min_cost = temp_row['total_blast_cost']
                best_explosive = explosive
                best_holes = holes
    print(f"Index {row.name}: Cost optimization - Predicted frags: {pred_frags}")
    if min_cost == float('inf'):
        print(f"Warning: No solution found for cost optimization at index {row.name}")
        min_cost = data.loc[row.name, 'total_blast_cost']
        best_holes = row['holes_blasted']
    return best_explosive, min_cost, best_holes, pred_frags

def optimize_fragmentation(row, model, features):
    burden_range = np.linspace(row['burden'] * 0.5, row['burden'] * 1.5, 10)
    spacing_range = np.linspace(row['spacing'] * 0.5, row['spacing'] * 1.5, 10)
    explosive_range = np.linspace(row['total_explosive_kg'] * 0.5, row['total_explosive_kg'] * 1.5, 10)
    hole_depth_range = np.linspace(row['hole_depth'] * 0.5, row['hole_depth'] * 1.5, 10)
    max_frag = 0
    best_params = {
        'burden': row['burden'],
        'spacing': row['spacing'],
        'total_explosive_kg': row['total_explosive_kg'],
        'hole_depth': row['hole_depth']
    }
    pred_frags = []
    for burden in burden_range:
        for spacing in spacing_range:
            for explosive in explosive_range:
                for hole_depth in hole_depth_range:
                    temp_row = row.copy()
                    temp_row['burden'] = burden
                    temp_row['spacing'] = spacing
                    temp_row['total_explosive_kg'] = explosive
                    temp_row['hole_depth'] = hole_depth
                    temp_row['burden_spacing'] = burden * spacing
                    temp_row['depth_to_bench'] = hole_depth / temp_row['bench_height']
                    temp_row['explosive_per_hole'] = explosive / temp_row['holes_blasted']
                    temp_row['x50'] = compute_x50(temp_row)
                    temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
                    temp_row['drill_cost'] = hole_depth * temp_row['holes_blasted'] * 150
                    temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
                    pred_frag = model.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
                    pred_frags.append(pred_frag)
                    if pred_frag > max_frag:
                        max_frag = pred_frag
                        best_params = {
                            'burden': burden,
                            'spacing': spacing,
                            'total_explosive_kg': explosive,
                            'hole_depth': hole_depth
                        }
    print(f"Index {row.name}: Fragmentation optimization - Predicted frags: {pred_frags}")
    return max_frag, best_params, pred_frags

def optimize_safety(row, model_ppv, model_frag, features, data):
    explosive_range = np.linspace(row['total_explosive_kg'] * 0.5, row['total_explosive_kg'] * 1.5, 20)
    holes_range = np.linspace(max(1, row['holes_blasted'] * 0.5), row['holes_blasted'] * 1.5, 10, dtype=int)
    min_ppv = float('inf')
    best_explosive = row['total_explosive_kg']
    best_holes = row['holes_blasted']
    pred_frags = []
    pred_ppvs = []
    for explosive in explosive_range:
        for holes in holes_range:
            temp_row = row.copy()
            temp_row['total_explosive_kg'] = explosive
            temp_row['holes_blasted'] = holes
            temp_row['explosive_per_hole'] = explosive / holes if holes > 0 else 0
            temp_row['x50'] = compute_x50(temp_row)
            temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
            temp_row['drill_cost'] = temp_row['hole_depth'] * holes * 150
            temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
            pred_frag = model_frag.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
            pred_ppv = model_ppv.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
            pred_frags.append(pred_frag)
            pred_ppvs.append(pred_ppv)
            if pred_frag >= FRAG_THRESHOLD and pred_ppv < min_ppv:
                min_ppv = pred_ppv
                best_explosive = explosive
                best_holes = holes
    print(f"Index {row.name}: Safety optimization - Predicted frags: {pred_frags}, Predicted PPVs: {pred_ppvs}")
    if min_ppv == float('inf'):
        print(f"Warning: No solution found for safety optimization at index {row.name}")
        min_ppv = data.loc[row.name, 'ppv']
        best_holes = row['holes_blasted']
    return min_ppv, best_explosive, best_holes, pred_frags, pred_ppvs

# Apply optimizations for test set
optimized_costs = []
optimized_frags = []
optimized_ppvs = []
optimized_holes = []
for idx, row_idx in enumerate(X_test.index):
    row = X.loc[row_idx].copy()
    row.name = row_idx
    best_explosive, min_cost, best_holes, cost_pred_frags = optimize_cost(row, pipelines['frag_in_range'], features, data)
    max_frag, best_params, frag_pred_frags = optimize_fragmentation(row, pipelines['frag_in_range'], features)
    min_ppv, best_explosive_ppv, best_holes_ppv, safety_pred_frags, safety_pred_ppvs = optimize_safety(
        row, pipelines['ppv'], pipelines['frag_in_range'], features, data
    )
    optimized_costs.append(min_cost)
    optimized_frags.append(max_frag)
    optimized_ppvs.append(min_ppv)
    optimized_holes.append(best_holes)

# Save optimization results and data for static plots
joblib.dump({
    'results': results,
    'optimized_costs': optimized_costs,
    'optimized_frags': optimized_frags,
    'optimized_ppvs': optimized_ppvs,
    'optimized_holes': optimized_holes
}, 'static_plot_data_nw.joblib')

print("Models and preprocessor saved successfully.")