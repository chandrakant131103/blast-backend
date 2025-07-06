from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import glob
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# --- Features and Targets ---
features = [
    'burden', 'holedia', 'spacing', 'hole_depth', 'sremming_length', 'bench_height',
    'hole_angle', 'total_rows', 'hole_blasted', 'column_charge_density',
    'avg_column_charge_length', 'avg_col_weight', 'total_explosive_kg', 'rock_density',
    'burden_spacing', 'depth_to_bench', 'explosive_per_hole', 'x50', 'x80',
    'frag_out_of_range', 'drill_cost', 'explosive_cost', 'total_blast_cost', 'ppv_alert'
]
targets = ['frag_in_range', 'frag_over_size', 'ppv']

# --- Load and Preprocess Data ---
try:
    data = pd.read_csv('Mine_Swamp.csv')
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    # Feature engineering
    data['burden_spacing'] = data['burden'] * data['spacing']
    data['depth_to_bench'] = data['hole_depth'] / data['bench_height']
    data['explosive_per_hole'] = data['total_explosive_kg'] / data['hole_blasted']

    def compute_x50(row):
        B = row['burden']
        S = row['spacing']
        H = row['hole_depth']
        Q = row['total_explosive_kg']
        return 0 if Q == 0 else 0.2 * ((B * S * H) / Q) ** 0.8

    data['x50'] = data.apply(compute_x50, axis=1)
    data['x80'] = data['x50'] * (np.log(5)) ** (1 / 1.2)
    data['frag_out_of_range'] = data['frag_over_size'] + (100 - data['frag_in_range'] - data['frag_over_size'])
    data['drill_cost'] = data['hole_depth'] * data['hole_blasted'] * 150
    data['explosive_cost'] = data['total_explosive_kg'] * 100
    data['total_blast_cost'] = data['drill_cost'] + data['explosive_cost']
    data['ppv_alert'] = (data['ppv'] > np.log1p(5)).astype(int)
    data['cost'] = data['total_explosive_kg'] * 100
    data['optimized_cost'] = data['total_explosive_kg'] * 95
    data['cost_savings'] = data['cost'] - data['optimized_cost']
    data['unsafe_blast'] = data['ppv'] > np.log1p(10)

    # Clean data
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
    if 'production_ton_therotical' in data.columns:
        data = data[data['production_ton_therotical'] > 0]
    data['ppv'] = np.log1p(data['ppv'])
    unsafe_count = int(data['unsafe_blast'].sum())
except FileNotFoundError:
    print("Error: Mine_Swamp.csv not found")
    data = pd.DataFrame()
    unsafe_count = 0

# --- Train Models ---
if not data.empty:
    X = data[features]
    y = data[targets]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([('num', StandardScaler(), features)])
    pipelines = {}
    results = {}
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, None],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
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
        joblib.dump(best_model, f'rf_model_{target}.joblib')

        y_pred = best_model.predict(X_test)
        results[target] = {
            'test_r2': float(r2_score(y_test[target], y_pred)),
            'test_mae': float(mean_absolute_error(y_test[target], y_pred)),
            'y_test': y_test[target].tolist(),
            'y_pred': y_pred.tolist()
        }

    # Summary calculations
    summary = {
        'avg_pred_frag': float(np.mean(results['frag_in_range']['y_pred'])),
        'avg_actual_frag': float(np.mean(y_test['frag_in_range'])),
        'avg_pred_oversize': float(np.mean(results['frag_over_size']['y_pred'])),
        'avg_actual_oversize': float(np.mean(y_test['frag_over_size'])),
        'avg_pred_ppv': float(np.expm1(np.mean(results['ppv']['y_pred']))),
        'avg_actual_ppv': float(np.expm1(np.mean(y_test['ppv']))),
        'avg_cost': float(data['cost'].mean() / 1000),
        'cost_savings': float(data['cost_savings'].mean() / 1000),
        'unsafe_blasts': unsafe_count,
        'optimized_avg_cost': float(data['optimized_cost'].mean() / 1000),
        'optimized_avg_frag': float(np.mean(results['frag_in_range']['y_pred'])),
        'optimized_avg_ppv': float(np.expm1(np.mean(results['ppv']['y_pred'])))
    }
else:
    pipelines = {}
    results = {target: {'test_r2': 0.0, 'test_mae': 0.0, 'y_test': [], 'y_pred': []} for target in targets}
    summary = {key: 0.0 for key in [
        'avg_pred_frag', 'avg_actual_frag', 'avg_pred_oversize', 'avg_actual_oversize',
        'avg_pred_ppv', 'avg_actual_ppv', 'avg_cost', 'cost_savings',
        'optimized_avg_cost', 'optimized_avg_frag', 'optimized_avg_ppv'
    ]}
    summary['unsafe_blasts'] = 0

# Ensure static directory is writable
static_dir = 'static'
try:
    os.makedirs(static_dir, exist_ok=True)
    test_file = os.path.join(static_dir, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
except PermissionError:
    print(f"Error: No write permissions for {static_dir}")
    exit(1)

def generate_charts(predictions, input_data, timestamp):
    charts = []
    os.makedirs(static_dir, exist_ok=True)

    # Clean up old charts
    old_files = glob.glob(os.path.join(static_dir, 'chart_*_*.png'))
    if len(old_files) > 10:
        for f in sorted(old_files, key=os.path.getmtime)[:-10]:
            os.remove(f)

    try:
        # Chart 1: Actual vs Predicted Fragmentation In Range
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=[input_data.get('frag_in_range', 0)], y=[predictions['frag_in_range']], alpha=0.6)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Input Fragmentation In Range (%)')
        plt.ylabel('Predicted Fragmentation In Range (%)')
        plt.title('Actual vs Predicted Fragmentation In Range')
        filename = f'chart_frag_in_range_{timestamp}.png'
        plt.savefig(os.path.join(static_dir, filename))
        plt.close()
        charts.append(filename)

        # Chart 2: Actual vs Predicted Costing
        input_cost = input_data.get('total_blast_cost', 0)
        optimized_cost = input_data.get('total_explosive_kg', 0) * 95
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Input', 'Predicted'], y=[input_cost / 1000, optimized_cost / 1000])
        plt.xlabel('Cost Type')
        plt.ylabel('Cost (₹1000)')
        plt.title('Actual vs Predicted Costing')
        filename = f'chart_costing_{timestamp}.png'
        plt.savefig(os.path.join(static_dir, filename))
        plt.close()
        charts.append(filename)

        # Chart 3: Actual vs Predicted PPV Safety
        plt.figure(figsize=(8, 6))
        input_ppv = np.expm1(input_data.get('ppv', 0))  # Inverse-transform input PPV
        pred_ppv = predictions['ppv']  # Already inverse-transformed
        sns.scatterplot(x=[input_ppv], y=[pred_ppv], alpha=0.6)
        max_val = max(input_ppv, pred_ppv) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Input PPV (mm/s)')
        plt.ylabel('Predicted PPV (mm/s)')
        plt.title('Actual vs Predicted PPV Safety')
        filename = f'chart_ppv_safety_{timestamp}.png'
        plt.savefig(os.path.join(static_dir, filename))
        plt.close()
        charts.append(filename)

        # Chart 4: Sample Index (Single Prediction)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=[1], y=[predictions['frag_in_range']], label='Predicted')
        if 'frag_in_range' in input_data:
            sns.scatterplot(x=[1], y=[input_data['frag_in_range']], label='Input')
        plt.xlabel('Sample Index')
        plt.ylabel('Fragmentation In Range (%)')
        plt.title('Fragmentation In Range vs Sample Index')
        plt.legend()
        filename = f'chart_sample_index_{timestamp}.png'
        plt.savefig(os.path.join(static_dir, filename))
        plt.close()
        charts.append(filename)

        # Chart 5: Correlation Heatmap
        plt.figure(figsize=(10, 8))
        input_df = pd.DataFrame([input_data])
        numeric_cols = ['burden', 'spacing', 'total_explosive_kg', 'frag_in_range', 'ppv']
        input_df['ppv'] = predictions['ppv']  # Use predicted PPV (already inverse-transformed)
        input_df['frag_in_range'] = predictions['frag_in_range'] if 'frag_in_range' not in input_data else input_data['frag_in_range']
        input_df = input_df[numeric_cols].fillna(0)
        corr_matrix = input_df.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f', annot_kws={'size': 12})
        plt.title('Correlation Heatmap for Input Parameters')
        filename = f'chart_heatmap_{timestamp}.png'
        plt.savefig(os.path.join(static_dir, filename))
        plt.close()
        charts.append(filename)
    except Exception as e:
        print(f"Error generating charts: {str(e)}")

    return charts

@app.route('/api/summary', methods=['GET'])
def get_summary():
    try:
        initial_charts = [
            'fragmentation_in_range.png',
            'fragmentation_over_size.png',
            'ppv_plot.png',
            'cost_plot.png',
            'safety_plot.png',
            'correlation_heatmap.png',
            'optimization_results.png',
            'frag_in_range_vs_index.png',
            'frag_out_of_range_vs_log10.png',
            'frag_in_range_actual_vs_predicted.png'
        ]
        existing_charts = [f for f in initial_charts if os.path.exists(os.path.join('static', f))]
        return jsonify({
            'summary': summary,
            'results': {k: {sk: float(sv) if isinstance(sv, (np.floating, np.integer)) else sv for sk, sv in v.items()}
                        for k, v in results.items()},
            'unsafe_count': unsafe_count,
            'charts': existing_charts
        })
    except Exception as e:
        print(f"Error in get_summary: {str(e)}")
        return jsonify({'error': f'Failed to fetch summary: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        # Validate required features
        required_features = ['burden', 'spacing', 'hole_depth', 'bench_height', 'total_explosive_kg', 'hole_blasted']
        missing_features = [f for f in required_features if f not in input_data]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Create DataFrame with input data
        df = pd.DataFrame([input_data])

        # Feature engineering
        df['burden_spacing'] = df['burden'] * df['spacing']
        df['depth_to_bench'] = df['hole_depth'] / df['bench_height']
        df['explosive_per_hole'] = df['total_explosive_kg'] / df['hole_blasted']
        df['x50'] = df.apply(compute_x50, axis=1)
        df['x80'] = df['x50'] * (np.log(5)) ** (1 / 1.2)
        df['frag_out_of_range'] = df.get('frag_over_size', 0) + (
                    100 - df.get('frag_in_range', 0) - df.get('frag_over_size', 0))
        df['drill_cost'] = df['hole_depth'] * df['hole_blasted'] * 150
        df['explosive_cost'] = df['total_explosive_kg'] * 100
        df['total_blast_cost'] = df['drill_cost'] + df['explosive_cost']
        df['ppv_alert'] = (df.get('ppv', 0) > np.log1p(5)).astype(int)

        # Ensure all features are present
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        # Predictions
        if not pipelines:
            return jsonify({'error': 'Models not loaded. Ensure training data is available.'}), 500

        predictions = {}
        for target in targets:
            pred = pipelines[target].predict(df[features])[0]
            if target == 'ppv':
                predictions[target] = float(np.expm1(pred))  # Inverse-transform PPV
            else:
                predictions[target] = float(pred)

        # Update summary
        updated_summary = {
            'avg_pred_frag': float(predictions['frag_in_range']),
            'avg_actual_frag': float(input_data.get('frag_in_range', summary['avg_actual_frag'])),
            'avg_pred_oversize': float(predictions['frag_over_size']),
            'avg_actual_oversize': float(input_data.get('frag_over_size', summary['avg_actual_oversize'])),
            'avg_pred_ppv': float(predictions['ppv']),
            'avg_actual_ppv': float(np.expm1(input_data.get('ppv', np.log1p(summary['avg_actual_ppv'])))),
            'avg_cost': float(df['total_blast_cost'].iloc[0] / 1000),
            'cost_savings': float((df['total_blast_cost'].iloc[0] - df['total_explosive_kg'].iloc[0] * 95) / 1000),
            'unsafe_blasts': int(predictions['ppv'] > 10),
            'optimized_avg_cost': float(df['total_explosive_kg'].iloc[0] * 95 / 1000),
            'optimized_avg_frag': float(predictions['frag_in_range']),
            'optimized_avg_ppv': float(predictions['ppv'])
        }

        # Generate dynamic charts
        timestamp = int(time.time())
        charts = generate_charts(predictions, input_data, timestamp)

        return jsonify({
            'summary': updated_summary,
            'results': results,
            'charts': charts
        })
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('static', filename)
    except FileNotFoundError:
        return jsonify({'error': 'Chart not found'}), 404

if __name__ == '__main__':
    if not data.empty:
        try:
            # Plot 1: Actual vs Predicted Fragmentation In Range
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=results['frag_in_range']['y_test'], y=results['frag_in_range']['y_pred'], alpha=0.6)
            plt.plot([min(results['frag_in_range']['y_test']), max(results['frag_in_range']['y_test'])],
                     [min(results['frag_in_range']['y_test']), max(results['frag_in_range']['y_test'])], 'r--')
            plt.xlabel('Actual Fragmentation In Range (%)')
            plt.ylabel('Predicted Fragmentation In Range (%)')
            plt.title(f'Fragmentation In Range: R²={results["frag_in_range"]["test_r2"]:.2f}')
            plt.savefig(os.path.join(static_dir, 'fragmentation_in_range.png'))
            plt.close()

            # Plot 2: Actual vs Predicted Fragmentation Over Size
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=results['frag_over_size']['y_test'], y=results['frag_over_size']['y_pred'], alpha=0.6)
            plt.plot([min(results['frag_over_size']['y_test']), max(results['frag_over_size']['y_test'])],
                     [min(results['frag_over_size']['y_test']), max(results['frag_over_size']['y_test'])], 'r--')
            plt.xlabel('Actual Fragmentation Over Size (%)')
            plt.ylabel('Predicted Fragmentation Over Size (%)')
            plt.title(f'Fragmentation Over Size: R²={results["frag_over_size"]["test_r2"]:.2f}')
            plt.savefig(os.path.join(static_dir, 'fragmentation_over_size.png'))
            plt.close()

            # Plot 3: Actual vs Predicted PPV
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=np.expm1(results['ppv']['y_test']), y=np.expm1(results['ppv']['y_pred']), alpha=0.6)
            plt.plot([min(np.expm1(results['ppv']['y_test'])), max(np.expm1(results['ppv']['y_test']))],
                     [min(np.expm1(results['ppv']['y_test'])), max(np.expm1(results['ppv']['y_test']))], 'r--')
            plt.xlabel('Actual PPV (mm/s)')
            plt.ylabel('Predicted PPV (mm/s)')
            plt.title(f'PPV: R²={results["ppv"]["test_r2"]:.2f}')
            plt.savefig(os.path.join(static_dir, 'ppv_plot.png'))
            plt.close()

            # Plot 4: Actual vs Optimized Cost
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data.loc[X_test.index, 'total_blast_cost'], y=data.loc[X_test.index, 'optimized_cost'],
                            alpha=0.6)
            plt.plot([data['total_blast_cost'].min(), data['total_blast_cost'].max()],
                     [data['total_blast_cost'].min(), data['total_blast_cost'].max()], 'r--')
            plt.xlabel('Actual Total Blast Cost (₹)')
            plt.ylabel('Optimized Total Blast Cost (₹)')
            plt.title('Actual vs Optimized Cost')
            plt.savefig(os.path.join(static_dir, 'cost_plot.png'))
            plt.close()

            # Plot 5: Safety (Confusion Matrix for PPV Alert)
            plt.figure(figsize=(8, 6))
            y_pred_ppv_alert = (np.array(results['ppv']['y_pred']) > np.log1p(5)).astype(int)
            cm = confusion_matrix(data.loc[X_test.index, 'ppv_alert'], y_pred_ppv_alert)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Danger'])
            disp.plot(cmap='Blues')
            plt.title('Safety: Actual vs Predicted PPV Alert')
            plt.savefig(os.path.join(static_dir, 'safety_plot.png'))
            plt.close()

            # Plot 6: Correlation Heatmap
            plt.figure(figsize=(10, 8))
            numeric_cols = ['burden', 'spacing', 'total_explosive_kg', 'frag_in_range', 'ppv']
            data_plot = data[numeric_cols].copy()
            data_plot['ppv'] = np.expm1(data_plot['ppv'])  # Inverse-transform PPV
            corr_matrix = data_plot.corr()
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f', annot_kws={'size': 12})
            plt.title('Correlation Heatmap')
            plt.savefig(os.path.join(static_dir, 'correlation_heatmap.png'))
            plt.close()

            # Plot 7: Optimization Results
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            sns.barplot(x=['Baseline', 'Optimized'], y=[data.loc[X_test.index, 'total_blast_cost'].mean() / 1000,
                                                        data.loc[X_test.index, 'optimized_cost'].mean() / 1000])
            plt.title('Average Cost (₹1000)')
            plt.subplot(1, 3, 2)
            sns.barplot(x=['Baseline', 'Optimized'],
                        y=[np.mean(results['frag_in_range']['y_test']), np.mean(results['frag_in_range']['y_pred'])])
            plt.title('Average Fragmentation In Range (%)')
            plt.subplot(1, 3, 3)
            sns.barplot(x=['Baseline', 'Optimized'],
                        y=[np.expm1(np.mean(results['ppv']['y_test'])), np.expm1(np.mean(results['ppv']['y_pred']))])
            plt.title('Average PPV (mm/s)')
            plt.savefig(os.path.join(static_dir, 'optimization_results.png'))
            plt.close()

            # Plot 8: Actual Frag In Range vs Sample Index
            plt.figure(figsize=(8, 6))
            sns.lineplot(x=range(len(results['frag_in_range']['y_test'])), y=results['frag_in_range']['y_test'])
            plt.xlabel('Sample Index')
            plt.ylabel('Actual Fragmentation In Range (%)')
            plt.title('Actual Fragmentation In Range vs Sample Index')
            plt.savefig(os.path.join(static_dir, 'frag_in_range_vs_index.png'))
            plt.close()

            # Plot 9: Actual Frag Out of Range vs Log10(x50)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=np.log10(data.loc[X_test.index, 'x50']), y=data.loc[X_test.index, 'frag_out_of_range'],
                            alpha=0.6)
            plt.xlabel('Log10(x50) (Fragment Size)')
            plt.ylabel('Actual Fragmentation Out of Range (%)')
            plt.title('Fragmentation Out of Range vs Log10(x50)')
            plt.savefig(os.path.join(static_dir, 'frag_out_of_range_vs_log10.png'))
            plt.close()

            # Plot 10: Actual vs Predicted Frag In Range vs Sample Index
            plt.figure(figsize=(8, 6))
            sns.lineplot(x=range(len(results['frag_in_range']['y_test'])), y=results['frag_in_range']['y_test'],
                         label='Actual')
            sns.lineplot(x=range(len(results['frag_in_range']['y_pred'])), y=results['frag_in_range']['y_pred'],
                         label='Predicted')
            plt.xlabel('Sample Index')
            plt.ylabel('Fragmentation In Range (%)')
            plt.title('Actual vs Predicted Fragmentation In Range vs Sample Index')
            plt.legend()
            plt.savefig(os.path.join(static_dir, 'frag_in_range_actual_vs_predicted.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating initial charts: {str(e)}")

    app.run(debug=True, port=5000)