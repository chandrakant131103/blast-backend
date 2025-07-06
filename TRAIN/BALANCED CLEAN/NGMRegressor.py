import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import differential_evolution
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['blastdate'])
    df = df.dropna(
        subset=['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'frag_in_range', 'frag_over_size', 'ppv',
                'total_drill_mtr', 'rock_name'])
    df = df[df['burden'] > 0]
    df = df[df['spacing'] > 0]
    df = df[df['hole_depth'] > 0]
    df = df[df['total_explosive_kg'] > 0]
    numeric_cols = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'rock_density',
                    'frag_in_range', 'frag_over_size', 'frag_under_size', 'ppv', 'flyrock',
                    'total_drill_mtr', 'total_exp_cost', 'drilling_cost']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if df['burden'].mean() > 100:
        df['burden'] /= 100
        df['spacing'] /= 100
        df['hole_depth'] /= 100
    le = LabelEncoder()
    df['rock_name_encoded'] = le.fit_transform(df['rock_name'])
    # Reset index to avoid indexing issues
    df = df.reset_index(drop=True)
    return df, le


# Train LightGBM models with train-test split
def train_models(df):
    features = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'rock_density', 'rock_name_encoded']
    X = df[features]
    targets = ['frag_in_range', 'frag_over_size', 'ppv']

    # Train-test split
    X_train, X_test, train_idx, test_idx = train_test_split(
        X, np.arange(len(X)), test_size=0.2, random_state=42, stratify=df['rock_name_encoded']
    )

    models = {}
    metrics = {'train_r2': {}, 'test_r2': {}, 'train_mae': {}, 'test_mae': {}}

    for target in targets:
        y = df[target]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        lgbm = LGBMRegressor(num_leaves=20, n_estimators=50, min_child_samples=10, random_state=42, n_jobs=-1)
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae')
        # For modern LightGBM (uncomment if version ≥ 3.0):
        # from lightgbm.callback import early_stopping
        # lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae',
        #          verbose_eval=False, callbacks=[early_stopping(stopping_rounds=10)])

        y_train_pred = lgbm.predict(X_train)
        y_test_pred = lgbm.predict(X_test)

        metrics['train_r2'][target] = r2_score(y_train, y_train_pred)
        metrics['test_r2'][target] = r2_score(y_test, y_test_pred)
        metrics['train_mae'][target] = mean_absolute_error(y_train, y_train_pred)
        metrics['test_mae'][target] = mean_absolute_error(y_test, y_test_pred)

        models[target] = lgbm

        # Print feature importance
        print(f"\nFeature Importance for {target}:")
        for feat, imp in zip(features, lgbm.feature_importances_):
            print(f"  {feat}: {imp}")

    return models, metrics, train_idx, test_idx, X_train, X_test


# Cost calculation
def calculate_costs(total_drill_mtr, explosive_weight, burden, spacing, orig_burden, orig_spacing, drill_cost_per_m=500,
                    exp_cost_per_kg=100):
    drill_mtr = total_drill_mtr * (burden / orig_burden) * (spacing / orig_spacing)
    drilling_cost = drill_mtr * drill_cost_per_m
    blasting_cost = explosive_weight * exp_cost_per_kg
    return drilling_cost + blasting_cost


# Optimization objective
def objective(params, row, models, target_frag=80, max_oversize=5, regulatory_ppv=10):
    burden, spacing, explosive_weight = params
    features = np.array(
        [[burden, spacing, row['hole_depth'], explosive_weight, row['rock_density'], row['rock_name_encoded']]])
    frag_in_range = models['frag_in_range'].predict(features)[0]
    frag_over_size = models['frag_over_size'].predict(features)[0]
    ppv = models['ppv'].predict(features)[0]
    total_cost = calculate_costs(row['total_drill_mtr'], explosive_weight, burden, spacing, row['burden'],
                                 row['spacing'])
    frag_penalty = max(0, target_frag - frag_in_range) * 1000
    oversize_penalty = max(0, frag_over_size - max_oversize) * 1000
    ppv_penalty = max(0, ppv - regulatory_ppv) * 1000
    return total_cost + frag_penalty + oversize_penalty + ppv_penalty


# Optimize blast parameters
def optimize_blast(row, models):
    bounds = [(row['burden'] * 0.7, row['burden'] * 1.3),
              (row['spacing'] * 0.7, row['spacing'] * 1.3),
              (row['total_explosive_kg'] * 0.7, row['total_explosive_kg'] * 1.3)]
    result = differential_evolution(
        objective,  # Use named function directly
        bounds,
        args=(row, models),
        maxiter=50,
        popsize=15,
        strategy='best1bin',
        workers=1  # Disable parallel processing
    )
    return result.x if result.success else [row['burden'], row['spacing'], row['total_explosive_kg']]


# Process train and test data
def process_blasts(file_path):
    df, le = load_data(file_path)
    models, metrics, train_idx, test_idx, X_train, X_test = train_models(df)

    # Initialize columns
    for split, idx, X in zip(['train', 'test'], [train_idx, test_idx], [X_train, X_test]):
        df.loc[idx, 'predicted_frag_in_range'] = models['frag_in_range'].predict(X)
        df.loc[idx, 'predicted_oversize'] = models['frag_over_size'].predict(X)
        df.loc[idx, 'predicted_ppv'] = models['ppv'].predict(X)
        df.loc[idx, 'total_cost'] = df.loc[idx].apply(
            lambda row: calculate_costs(row['total_drill_mtr'], row['total_explosive_kg'], row['burden'],
                                        row['spacing'], row['burden'], row['spacing']),
            axis=1
        )
        df.loc[idx, 'optimized_burden'] = 0.0
        df.loc[idx, 'optimized_spacing'] = 0.0
        df.loc[idx, 'optimized_explosive'] = 0.0
        df.loc[idx, 'optimized_cost'] = 0.0

    # Optimize blasts
    for idx in np.concatenate([train_idx, test_idx]):
        row = df.loc[idx]
        opt_params = optimize_blast(row, models)
        df.at[idx, 'optimized_burden'] = opt_params[0]
        df.at[idx, 'optimized_spacing'] = opt_params[1]
        df.at[idx, 'optimized_explosive'] = opt_params[2]
        df.at[idx, 'optimized_cost'] = calculate_costs(row['total_drill_mtr'], opt_params[2], opt_params[0],
                                                       opt_params[1], row['burden'], row['spacing'])
        print(
            f"Blast {row['blastcode']}: Predicted In-Range {df.at[idx, 'predicted_frag_in_range']:.2f}% vs Actual {row['frag_in_range']:.2f}%, "
            f"Oversize {df.at[idx, 'predicted_oversize']:.2f}% vs Actual {row['frag_over_size']:.2f}%, "
            f"Cost ${df.at[idx, 'total_cost'] / 1000:.2f}K vs Optimized ${df.at[idx, 'optimized_cost'] / 1000:.2f}K")

    # Save indices
    np.save('train_indices.npy', train_idx)
    np.save('test_indices.npy', test_idx)

    return df, metrics, train_idx, test_idx


# Visualization functions
def plot_fragmentation_curve(df, split, idx):
    plt.figure(figsize=(10, 6))
    subset = df.loc[idx]
    plt.scatter(subset['predicted_frag_in_range'], subset['frag_in_range'], c='blue', alpha=0.5, label='In-Range')
    plt.scatter(subset['predicted_oversize'], subset['frag_over_size'], c='red', alpha=0.5, label='Oversize')
    plt.plot([0, 100], [0, 100], color='black', linestyle='--')
    plt.xlabel('Predicted (%)')
    plt.ylabel('Actual (%)')
    plt.title(f'Predicted vs Actual Fragmentation ({split.capitalize()})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'fragmentation_curve_{split}.png')
    plt.close()


def plot_cost_savings(df, split, idx):
    plt.figure(figsize=(10, 6))
    subset = df.loc[idx]
    cost_savings = (subset['total_cost'] - subset['optimized_cost']) / 1000
    plt.scatter(subset['total_cost'] / 1000, cost_savings, c='green', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Actual Cost ($K)')
    plt.ylabel('Cost Savings ($K)')
    plt.title(f'Cost Savings vs Actual Cost ({split.capitalize()})')
    plt.grid(True)
    plt.savefig(f'cost_savings_vs_actual_{split}.png')
    plt.close()


def plot_oversize_comparison(df, split, idx):
    plt.figure(figsize=(10, 6))
    subset = df.loc[idx]
    plt.scatter(subset['predicted_oversize'], subset['frag_over_size'], c='purple', alpha=0.5)
    plt.plot([0, 100], [0, 100], color='black', linestyle='--')
    plt.xlabel('Predicted Oversize (%)')
    plt.ylabel('Actual Oversize (%)')
    plt.title(f'Predicted vs Actual Oversize ({split.capitalize()})')
    plt.grid(True)
    plt.savefig(f'oversize_comparison_{split}.png')
    plt.close()


def plot_safety_bar(df, split, idx, regulatory_ppv=10):
    plt.figure(figsize=(12, 6))
    subset = df.loc[idx].head(20)
    bar_width = 0.35
    index = np.arange(len(subset))
    plt.bar(index, subset['ppv'], bar_width, label='Actual PPV', color='teal')
    plt.bar(index + bar_width, subset['predicted_ppv'], bar_width, label='Predicted PPV', color='orange')
    plt.axhline(y=regulatory_ppv, color='red', linestyle='--', label='Regulatory PPV')
    plt.xlabel('Blast Code')
    plt.ylabel('PPV (mm/s)')
    plt.title(f'Safety Compliance (PPV) ({split.capitalize()})')
    plt.xticks(index + bar_width / 2, subset['blastcode'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'safety_bar_{split}.png')
    plt.close()


# Main execution
if __name__ == '__main__':
    file_path = 'C:\\Users\\fr16c\\PyCharmMiscProject\\your_cleaned_file.csv'
    df, metrics, train_idx, test_idx = process_blasts(file_path)

    # Split data for reporting
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    # Summary stats
    for split, df_split in [('train', df_train), ('test', df_test)]:
        print(f"\n{split.capitalize()} Set Summary")
        print(f"Average Predicted Fragmentation (% in-range): {df_split['predicted_frag_in_range'].mean():.2f}%")
        print(f"Average Actual Fragmentation (% in-range): {df_split['frag_in_range'].mean():.2f}%")
        print(f"Average Predicted Oversize: {df_split['predicted_oversize'].mean():.2f}%")
        print(f"Average Actual Oversize: {df_split['frag_over_size'].mean():.2f}%")
        print(f"Average Original Cost: ${df_split['total_cost'].mean() / 1000:.2f}K")
        print(f"Average Optimized Cost: ${df_split['optimized_cost'].mean() / 1000:.2f}K")
        print(
            f"Average Cost Savings: ${(df_split['total_cost'] - df_split['optimized_cost']).mean() / 1000:.2f}K per blast")
        print(f"Safety Violations (PPV > 10 mm/s): {len(df_split[df_split['ppv'] > 10])}")

    print("\nModel Metrics:")
    print(
        f"  Fragmentation (In-Range): Train R²={metrics['train_r2']['frag_in_range']:.2f}, Test R²={metrics['test_r2']['frag_in_range']:.2f}, "
        f"Train MAE={metrics['train_mae']['frag_in_range']:.2f}%, Test MAE={metrics['test_mae']['frag_in_range']:.2f}%")
    print(
        f"  Oversize: Train R²={metrics['train_r2']['frag_over_size']:.2f}, Test R²={metrics['test_r2']['frag_over_size']:.2f}, "
        f"Train MAE={metrics['train_mae']['frag_over_size']:.2f}%, Test MAE={metrics['test_mae']['frag_over_size']:.2f}%")
    print(f"  PPV: Train R²={metrics['train_r2']['ppv']:.2f}, Test R²={metrics['test_r2']['ppv']:.2f}, "
          f"Train MAE={metrics['train_mae']['ppv']:.2f} mm/s, Test MAE={metrics['test_mae']['ppv']:.2f} mm/s")

    # Blast 2648 analysis
    for split, df_split in [('train', df_train), ('test', df_test)]:
        blast_2648 = df_split[df_split['blastcode'] == '2648']
        if not blast_2648.empty:
            row = blast_2648.iloc[0]
            print(
                f"\nBlast 2648 ({split.capitalize()}): Predicted In-Range {row['predicted_frag_in_range']:.2f}% vs Actual {row['frag_in_range']:.2f}%, "
                f"Oversize {row['predicted_oversize']:.2f}% vs Actual {row['frag_over_size']:.2f}%, "
                f"Cost ${row['total_cost'] / 1000:.2f}K vs Optimized ${row['optimized_cost'] / 1000:.2f}K, "
                f"Burden={row['burden']:.2f}m, Spacing={row['spacing']:.2f}m")

    # Interesting fact
    best_blast = df.loc[np.concatenate([train_idx, test_idx]), 'frag_in_range'].idxmax()
    best_split = 'train' if best_blast in train_idx else 'test'
    best_row = df.loc[best_blast]
    print(
        f"\nInteresting Fact: Blast {best_row['blastcode']} ({best_split.capitalize()}) achieved {best_row['frag_in_range']:.2f}% in-range "
        f"with oversize={best_row['frag_over_size']:.2f}%, burden={best_row['optimized_burden']:.2f}m, "
        f"spacing={best_row['optimized_spacing']:.2f}m, "
        f"cost savings=${(best_row['total_cost'] - best_row['optimized_cost']) / 1000:.2f}K")

    # Save results
    cols = ['blastcode', 'burden', 'optimized_burden', 'spacing', 'optimized_spacing',
            'total_explosive_kg', 'optimized_explosive', 'total_cost', 'optimized_cost',
            'predicted_frag_in_range', 'frag_in_range', 'predicted_oversize', 'frag_over_size',
            'predicted_ppv', 'ppv']
    df_train[cols].to_csv('optimized_blast_patterns_train.csv', index=False)
    df_test[cols].to_csv('optimized_blast_patterns_test.csv', index=False)

    # Generate visualizations
    for split, idx in [('train', train_idx), ('test', test_idx)]:
        plot_fragmentation_curve(df, split, idx)
        plot_cost_savings(df, split, idx)
        plot_oversize_comparison(df, split, idx)
        plot_safety_bar(df, split, idx)

    print("\nPlots saved: fragmentation_curve_[train/test].png, cost_savings_vs_actual_[train/test].png, "
          "oversize_comparison_[train/test].png, safety_bar_[train/test].png")
    print("Optimized patterns saved to 'optimized_blast_patterns_train.csv' and 'optimized_blast_patterns_test.csv'")