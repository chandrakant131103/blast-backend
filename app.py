import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import time
import glob

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://your-react-app.onrender.com"]}})

# Load pre-trained models and preprocessor
pipelines = {
    'frag_in_range': joblib.load('rf_model_frag_in_range.joblib'),
    'frag_over_size': joblib.load('rf_model_frag_over_size.joblib'),
    'ppv': joblib.load('rf_model_ppv.joblib')
}
preprocessor = joblib.load('preprocessor.joblib')

# Load dataset for static plots and optimization
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

# Load test indices and static plot data
X_test_indices = joblib.load('X_test_indices.joblib')
static_plot_data = joblib.load('static_plot_data_nw.joblib')
results = static_plot_data['results']
optimized_costs = static_plot_data['optimized_costs']
optimized_frags = static_plot_data['optimized_frags']
optimized_ppvs = static_plot_data['optimized_ppvs']
optimized_holes = static_plot_data['optimized_holes']

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
        min_cost = data.loc[row.name, 'total_blast_cost'] if row.name in data.index else row['total_blast_cost']
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
        min_ppv = data.loc[row.name, 'ppv'] if row.name in data.index else row['ppv'] if 'ppv' in row else float('inf')
        best_holes = row['holes_blasted']
    return min_ppv, best_explosive, best_holes, pred_frags, pred_ppvs

def analyze_parameter_impact(row, model_frag, model_ppv, features):
    parameters = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'holes_blasted']
    ranges = {
        'burden': np.linspace(row['burden'] * 0.5, row['burden'] * 1.5, 10),
        'spacing': np.linspace(row['spacing'] * 0.5, row['spacing'] * 1.5, 10),
        'hole_depth': np.linspace(row['hole_depth'] * 0.5, row['hole_depth'] * 1.5, 10),
        'total_explosive_kg': np.linspace(row['total_explosive_kg'] * 0.5, row['total_explosive_kg'] * 1.5, 10),
        'holes_blasted': np.linspace(max(1, row['holes_blasted'] * 0.5), row['holes_blasted'] * 1.5, 5, dtype=int)
    }
    impact_data = {param: {'frag': [], 'ppv': [], 'cost': []} for param in parameters}
    for param in parameters:
        for value in ranges[param]:
            temp_row = row.copy()
            temp_row[param] = value
            if param in ['burden', 'spacing']:
                temp_row['burden_spacing'] = temp_row['burden'] * temp_row['spacing']
            if param == 'hole_depth':
                temp_row['depth_to_bench'] = temp_row['hole_depth'] / temp_row['bench_height']
                temp_row['drill_cost'] = temp_row['hole_depth'] * temp_row['holes_blasted'] * 150
                temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
            if param == 'total_explosive_kg':
                temp_row['explosive_per_hole'] = temp_row['total_explosive_kg'] / temp_row['holes_blasted']
                temp_row['explosive_cost'] = temp_row['total_explosive_kg'] * 100
                temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
            if param == 'holes_blasted':
                temp_row['explosive_per_hole'] = temp_row['total_explosive_kg'] / temp_row['holes_blasted']
                temp_row['drill_cost'] = temp_row['hole_depth'] * temp_row['holes_blasted'] * 150
                temp_row['total_blast_cost'] = temp_row['drill_cost'] + temp_row['explosive_cost']
            temp_row['x50'] = compute_x50(temp_row)
            temp_row['x80'] = temp_row['x50'] * (np.log(5)) ** (1 / 1.2)
            pred_frag = model_frag.predict(pd.DataFrame([temp_row[features]], columns=features))[0]
            pred_ppv = np.expm1(model_ppv.predict(pd.DataFrame([temp_row[features]], columns=features))[0])
            impact_data[param]['frag'].append(pred_frag)
            impact_data[param]['ppv'].append(pred_ppv)
            impact_data[param]['cost'].append(temp_row['total_blast_cost'])
    return impact_data, ranges

def clean_plot_directory():
    plot_dir = 'plots'
    for plot_file in glob.glob(f'{plot_dir}/dynamic_*.png'):
        os.remove(plot_file)

def generate_dynamic_plots(input_df, pred_frag_in_range, pred_ppv, opt_frag, opt_ppv, opt_holes, actual_holes, impact_data, ranges):
    plt.style.use('default')
    sns.set_style('whitegrid')
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_files = []
    timestamp = int(time.time())

    # Plot 1: Current vs Optimized
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=[pred_frag_in_range], y=[pred_ppv], s=100, alpha=0.8, label='Current')
    sns.scatterplot(x=[opt_frag], y=[opt_ppv], s=100, alpha=0.8, label='Optimized')
    plt.xlabel('Fragmentation In Range (%)')
    plt.ylabel('PPV (mm/s)')
    plt.title('Current vs Optimized Results')
    plt.legend()
    plot_path = f'{plot_dir}/dynamic_prediction_plot_{timestamp}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 2: Actual vs Optimized Number of Holes
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Actual', 'Optimized'], y=[actual_holes, opt_holes])
    plt.xlabel('Blast Configuration')
    plt.ylabel('Number of Holes')
    plt.title('Actual vs Optimized Number of Holes')
    plot_path = f'{plot_dir}/dynamic_holes_plot_{timestamp}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 3: Parameter Impact Analysis
    plt.figure(figsize=(15, 10))
    parameters = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'holes_blasted']
    for i, param in enumerate(parameters, 1):
        plt.subplot(3, 2, i)
        plt.plot(ranges[param], impact_data[param]['frag'], label='Fragmentation (%)', color='blue')
        plt.plot(ranges[param], impact_data[param]['ppv'], label='PPV (mm/s)', color='orange')
        plt.plot(ranges[param], np.array(impact_data[param]['cost']) / 1000, label='Cost (₹1000)', color='green')
        plt.xlabel(param.replace('_', ' ').title())
        plt.title(f'Impact of {param.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plot_path = f'{plot_dir}/dynamic_impact_plot_{timestamp}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    return plot_files

def generate_static_plots(results, data, X_test_indices, optimized_costs, optimized_frags, optimized_ppvs, optimized_holes):
    plt.style.use('default')
    sns.set_style('whitegrid')
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_files = []

    # Plot 1: Actual vs Predicted frag_in_range
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=results['frag_in_range']['y_test'], y=results['frag_in_range']['y_pred'], alpha=0.6)
    plt.plot([results['frag_in_range']['y_test'].min(), results['frag_in_range']['y_test'].max()],
             [results['frag_in_range']['y_test'].min(), results['frag_in_range']['y_test'].max()], 'r--')
    plt.xlabel('Actual Fragmentation In Range (%)')
    plt.ylabel('Predicted Fragmentation In Range (%)')
    plt.title(f'Fragmentation In Range: R²={results["frag_in_range"]["test_r2"]:.2f}')
    plot_path = f'{plot_dir}/fragmentation_in_range.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 2: Actual vs Predicted frag_over_size
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=results['frag_over_size']['y_test'], y=results['frag_over_size']['y_pred'], alpha=0.6)
    plt.plot([results['frag_over_size']['y_test'].min(), results['frag_over_size']['y_test'].max()],
             [results['frag_over_size']['y_test'].min(), results['frag_over_size']['y_test'].max()], 'r--')
    plt.xlabel('Actual Fragmentation Over Size (%)')
    plt.ylabel('Predicted Fragmentation Over Size (%)')
    plt.title(f'Fragmentation Over Size: R²={results["frag_over_size"]["test_r2"]:.2f}')
    plot_path = f'{plot_dir}/fragmentation_over_size.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 3: Actual vs Predicted PPV
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=np.expm1(results['ppv']['y_test']), y=np.expm1(results['ppv']['y_pred']), alpha=0.6)
    plt.plot([np.expm1(results['ppv']['y_test']).min(), np.expm1(results['ppv']['y_test']).max()],
             [np.expm1(results['ppv']['y_test']).min(), np.expm1(results['ppv']['y_test']).max()], 'r--')
    plt.xlabel('Actual PPV (mm/s)')
    plt.ylabel('Predicted PPV (mm/s)')
    plt.title(f'PPV: R²={results["ppv"]["test_r2"]:.2f}')
    plot_path = f'{plot_dir}/ppv_plot.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 4: Actual vs Optimized Cost
    if len(optimized_costs) == len(X_test_indices):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data.loc[X_test_indices, 'total_blast_cost'], y=optimized_costs, alpha=0.6)
        plt.plot([data['total_blast_cost'].min(), data['total_blast_cost'].max()],
                 [data['total_blast_cost'].min(), data['total_blast_cost'].max()], 'r--')
        plt.xlabel('Actual Total Blast Cost (₹)')
        plt.ylabel('Optimized Total Blast Cost (₹)')
        plt.title('Actual vs Optimized Cost')
        plot_path = f'{plot_dir}/cost_plot.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_path)

    # Plot 5: Safety (Confusion Matrix for ppv_alert)
    plt.figure(figsize=(8, 6))
    y_pred_ppv_alert = (results['ppv']['y_pred'] > GROUND_VIBRATION_LIMIT).astype(int)
    cm = ConfusionMatrixDisplay.from_predictions(
        data.loc[X_test_indices, 'ppv_alert'], y_pred_ppv_alert, display_labels=['Safe', 'Danger']
    )
    cm.plot(ax=plt.gca(), cmap='Blues')
    plt.title('Safety: Actual vs Predicted PPV Alert')
    plot_path = f'{plot_dir}/safety_plot.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 6: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'holes_blasted', 'frag_in_range', 'ppv']
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plot_path = f'{plot_dir}/correlation_heatmap.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)

    # Plot 7: Optimization Results
    if len(optimized_costs) == len(X_test_indices):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.barplot(x=['Baseline', 'Optimized'],
                    y=[data.loc[X_test_indices, 'total_blast_cost'].mean() / 1000, np.mean(optimized_costs) / 1000])
        plt.title('Average Cost (₹1000)')
        plt.subplot(1, 3, 2)
        sns.barplot(x=['Baseline', 'Optimized'],
                    y=[results['frag_in_range']['y_test'].mean(), np.mean(optimized_frags)])
        plt.title('Average Fragmentation In Range (%)')
        plt.subplot(1, 3, 3)
        sns.barplot(x=['Baseline', 'Optimized'],
                    y=[np.expm1(results['ppv']['y_test']).mean(), np.expm1(np.mean(optimized_ppvs))])
        plt.title('Average PPV (mm/s)')
        plt.tight_layout()
        plot_path = f'{plot_dir}/optimization_results.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_path)

    # Plot 8: Actual vs Optimized Number of Holes
    if len(optimized_holes) == len(X_test_indices):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data.loc[X_test_indices, 'holes_blasted'], y=optimized_holes, alpha=0.6)
        plt.plot([data['holes_blasted'].min(), data['holes_blasted'].max()],
                 [data['holes_blasted'].min(), data['holes_blasted'].max()], 'r--')
        plt.xlabel('Actual Number of Holes')
        plt.ylabel('Optimized Number of Holes')
        plt.title('Actual vs Optimized Number of Holes')
        plot_path = f'{plot_dir}/holes_plot.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_path)

    return plot_files

# Generate static plots
plot_urls = generate_static_plots(results, data, X_test_indices, optimized_costs, optimized_frags, optimized_ppvs, optimized_holes)

@app.route('/api/optimize', methods=['POST', 'OPTIONS'])
def optimize():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        start_time = time.time()
        data = request.get_json()
        if not data or 'parameters' not in data:
            return jsonify({'error': 'No input parameters provided'}), 400

        input_data = data['parameters']
        input_df = pd.DataFrame([input_data])
        input_df.columns = input_df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Convert inputs to numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Check for missing or invalid inputs
        required_inputs = [
            'burden', 'hole_diameter', 'spacing', 'hole_depth', 'stemming_length',
            'bench_height', 'hole_angle', 'total_rows', 'holes_blasted',
            'column_charge_density', 'avg_column_charge_length', 'avg_column_weight',
            'total_explosive_kg', 'rock_density'
        ]
        missing_cols = [col for col in required_inputs if col not in input_df.columns or input_df[col].isna().any()]
        if missing_cols:
            return jsonify({'error': f'Missing or invalid inputs: {missing_cols}'}), 400

        # Handle zero hole_angle
        if input_df['hole_angle'].iloc[0] == 0:
            print("Warning: hole_angle is zero, using small non-zero value for stability")
            input_df['hole_angle'] = input_df['hole_angle'].replace(0, 1e-10)

        # Feature engineering
        input_df['burden_spacing'] = input_df['burden'] * input_df['spacing']
        input_df['depth_to_bench'] = input_df['hole_depth'] / input_df['bench_height']
        input_df['explosive_per_hole'] = input_df['total_explosive_kg'] / input_df['holes_blasted']
        input_df['x50'] = input_df.apply(compute_x50, axis=1)
        input_df['x80'] = input_df['x50'] * (np.log(5)) ** (1 / 1.2)
        input_df['frag_out_of_range'] = 0
        input_df['drill_cost'] = input_df['hole_depth'] * input_df['holes_blasted'] * 150
        input_df['explosive_cost'] = input_df['total_explosive_kg'] * 100
        input_df['total_blast_cost'] = input_df['drill_cost'] + input_df['explosive_cost']
        input_df['ppv_alert'] = 0

        # Ensure all features
        missing_features = [col for col in features if col not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        X = input_df[features]
        optimization_type = data.get('optimization_type', 'cost')

        # Predict
        pred_frag_in_range = pipelines['frag_in_range'].predict(X)[0]
        pred_frag_over_size = pipelines['frag_over_size'].predict(X)[0]
        pred_ppv = np.expm1(pipelines['ppv'].predict(X)[0])
        actual_holes = input_df['holes_blasted'].iloc[0]

        # Apply optimization
        row = input_df.iloc[0].copy()
        row.name = 0
        if optimization_type == 'cost':
            best_explosive, opt_cost, opt_holes, _ = optimize_cost(row, pipelines['frag_in_range'], features, input_df)
            opt_frag = pred_frag_in_range
            opt_ppv = pred_ppv
            best_params = {
                'burden': row['burden'],
                'spacing': row['spacing'],
                'hole_depth': row['hole_depth'],
                'total_explosive_kg': best_explosive,
                'holes_blasted': opt_holes
            }
        elif optimization_type == 'fragmentation':
            opt_frag, best_params, _ = optimize_fragmentation(row, pipelines['frag_in_range'], features)
            opt_cost = input_df['total_blast_cost'].iloc[0]
            opt_ppv = pred_ppv
            opt_holes = row['holes_blasted']
        else:  # safety
            opt_ppv, best_explosive, opt_holes, _, _ = optimize_safety(row, pipelines['ppv'], pipelines['frag_in_range'], features, input_df)
            opt_cost = input_df['total_blast_cost'].iloc[0]
            opt_frag = pred_frag_in_range
            best_params = {
                'burden': row['burden'],
                'spacing': row['spacing'],
                'hole_depth': row['hole_depth'],
                'total_explosive_kg': best_explosive,
                'holes_blasted': opt_holes
            }

        # Parameter impact analysis
        impact_data, ranges = analyze_parameter_impact(row, pipelines['frag_in_range'], pipelines['ppv'], features)

        # Clean old dynamic plots
        clean_plot_directory()

        # Generate dynamic plots
        dynamic_plot_urls = generate_dynamic_plots(input_df, pred_frag_in_range, pred_ppv, opt_frag, opt_ppv, opt_holes, actual_holes, impact_data, ranges)

        # Count unsafe blasts
        unsafe_blasts = 1 if pred_ppv > 5 else 0
        optimized_unsafe_blasts = 1 if opt_ppv > 5 else 0

        # Prepare response
        response = {
            'current': {
                'predictions': {
                    'frag_in_range': float(pred_frag_in_range),
                    'frag_over_size': float(pred_frag_over_size),
                    'ppv': float(pred_ppv)
                },
                'engineered': {
                    'total_blast_cost': float(input_df['total_blast_cost'].iloc[0]),
                    'burden_spacing': float(input_df['burden_spacing'].iloc[0]),
                    'explosive_per_hole': float(input_df['explosive_per_hole'].iloc[0]),
                    'holes_blasted': float(actual_holes)
                },
                'safety_rating': 'Good' if pred_ppv <= 5 else 'Caution',
                'unsafe_blasts': unsafe_blasts
            },
            'optimized': {
                'predictions': {
                    'frag_in_range': float(opt_frag),
                    'ppv': float(opt_ppv)
                },
                'engineered': {
                    'total_blast_cost': float(opt_cost),
                    'burden': float(best_params['burden']),
                    'spacing': float(best_params['spacing']),
                    'hole_depth': float(best_params['hole_depth']),
                    'total_explosive_kg': float(best_params['total_explosive_kg']),
                    'holes_blasted': float(opt_holes)
                },
                'safety_rating': 'Excellent' if opt_ppv <= 5 else 'Good',
                'unsafe_blasts': optimized_unsafe_blasts
            },
            'improvements': {
                'cost_savings': float(input_df['total_blast_cost'].iloc[0] - opt_cost),
                'cost_savings_percentage': float(
                    (input_df['total_blast_cost'].iloc[0] - opt_cost) / input_df['total_blast_cost'].iloc[0] * 100) if
                input_df['total_blast_cost'].iloc[0] != 0 else 0,
                'fragmentation_improvement': float(opt_frag - pred_frag_in_range),
                'safety_improvement': float(pred_ppv - opt_ppv),
                'holes_reduction': float(actual_holes - opt_holes)
            },
            'recommendations': [
                f'Adjust burden to {best_params["burden"]:.2f} m, spacing to {best_params["spacing"]:.2f} m, hole depth to {best_params["hole_depth"]:.2f} m, and explosive to {best_params["total_explosive_kg"]:.2f} kg for optimal results.',
                f'Monitor PPV to ensure safety compliance (optimized PPV: {opt_ppv:.2f} mm/s).',
                f'Optimize number of holes to {opt_holes:.0f} to reduce costs.'
            ],
            'plot_urls': [f'/plots/{os.path.basename(plot)}' for plot in plot_urls + dynamic_plot_urls],
            'parameter_impact': {
                param: {
                    'values': ranges[param].tolist(),
                    'fragmentation': impact_data[param]['frag'],
                    'ppv': impact_data[param]['ppv'],
                    'cost': impact_data[param]['cost']
                } for param in impact_data
            }
        }

        print(f"Request processed in {time.time() - start_time:.2f} seconds")
        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /api/optimize: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    plot_path = os.path.join(os.getcwd(), 'plots', filename)
    if os.path.exists(plot_path):
        response = send_file(plot_path, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return jsonify({'error': 'Plot not found'}), 404

if __name__ == '__main__':
    print("Loaded pre-trained models and preprocessor")
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug for better performance