import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# Load and preprocess CSV data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['blastdate'])
    df = df.dropna(subset=['burden', 'spacing', 'hole_depth', 'total_explosive_kg'])
    df = df[df['burden'] > 0]
    df = df[df['spacing'] > 0]
    df = df[df['hole_depth'] > 0]
    df = df[df['total_explosive_kg'] > 0]
    numeric_cols = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'rock_density',
                    'frag_in_range', 'frag_over_size', 'frag_under_size', 'ppv', 'flyrock',
                    'total_drill_mtr', 'total_exp_cost', 'drilling_cost']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Ensure units (convert cm to m if needed)
    if df['burden'].mean() > 100:  # Likely in cm
        df['burden'] /= 100
        df['spacing'] /= 100
        df['hole_depth'] /= 100
    return df


# Kuz-Ram model for fragmentation
def kuz_ram_x50(burden, spacing, hole_depth, explosive_weight, rock_type):
    A = 1.0 if 'COAL' in rock_type.upper() else 0.8
    x50 = A * (burden * spacing) ** 0.8 * hole_depth ** 0.1 * explosive_weight ** -0.3
    return max(0.01, min(x50, 1.0))  # Clamp x50 to realistic range


# Estimate fragment size distribution
def estimate_frag_distribution(x50, target_range=(0.01, 0.3)):
    print(f"x50: {x50}")  # Debug
    k = 2.2
    size_min, size_max = target_range
    p_min = 1 - np.exp(-((size_min / x50) ** k))
    p_max = 1 - np.exp(-((size_max / x50) ** k))
    frag_in_range = (p_max - p_min) * 100
    return max(0, min(frag_in_range, 100))  # Clamp to 0–100%


# Vibration prediction
def predict_ppv(charge_per_delay, distance=100, k=114, n=1.6):
    scaled_distance = distance / (charge_per_delay ** (1 / 3))
    ppv = k * (scaled_distance ** -n)
    return ppv


# Cost calculation
def calculate_costs(total_drill_mtr, explosive_weight, drill_cost_per_m=500, exp_cost_per_kg=100):
    drilling_cost = total_drill_mtr * drill_cost_per_m
    blasting_cost = explosive_weight * exp_cost_per_kg
    return drilling_cost + blasting_cost, drilling_cost, blasting_cost


# Optimization function
def optimize_blast(row, target_frag=80, regulatory_ppv=10, drill_cost_per_m=500, exp_cost_per_kg=100):
    def objective(params):
        burden, spacing, explosive_weight = params
        total_drill_mtr = row['total_drill_mtr'] * (burden / row['burden']) * (spacing / row['spacing'])
        cost, _, _ = calculate_costs(total_drill_mtr, explosive_weight, drill_cost_per_m, exp_cost_per_kg)
        return cost

    def frag_constraint(params):
        burden, spacing, explosive_weight = params
        x50 = kuz_ram_x50(burden, spacing, row['hole_depth'], explosive_weight, row['rock_name'])
        frag_in_range = estimate_frag_distribution(x50)
        return frag_in_range - target_frag

    def ppv_constraint(params):
        _, _, explosive_weight = params
        charge_per_delay = explosive_weight / (row['total_rows'] or 1)
        ppv = predict_ppv(charge_per_delay)
        return regulatory_ppv - ppv

    constraints = [
        {'type': 'ineq', 'fun': frag_constraint},
        {'type': 'ineq', 'fun': ppv_constraint}
    ]
    bounds = [(row['burden'] * 0.8, row['burden'] * 1.2),
              (row['spacing'] * 0.8, row['spacing'] * 1.2),
              (row['total_explosive_kg'] * 0.8, row['total_explosive_kg'] * 1.2)]
    initial_guess = [row['burden'], row['spacing'], row['total_explosive_kg']]
    result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds, method='SLSQP')
    return result.x if result.success else initial_guess


# Main processing function
def process_blasts(file_path):
    df = load_data(file_path)
    df['x50'] = 0.0
    df['predicted_frag_in_range'] = 0.0
    df['predicted_ppv'] = 0.0
    df['total_cost'] = 0.0
    df['optimized_burden'] = 0.0
    df['optimized_spacing'] = 0.0
    df['optimized_explosive'] = 0.0

    for idx, row in df.iterrows():
        x50 = kuz_ram_x50(row['burden'], row['spacing'], row['hole_depth'], row['total_explosive_kg'], row['rock_name'])
        frag_in_range = estimate_frag_distribution(x50)
        df.at[idx, 'x50'] = x50
        df.at[idx, 'predicted_frag_in_range'] = frag_in_range
        charge_per_delay = row['total_explosive_kg'] / (row['total_rows'] or 1)
        df.at[idx, 'predicted_ppv'] = predict_ppv(charge_per_delay)
        total_cost, _, _ = calculate_costs(row['total_drill_mtr'], row['total_explosive_kg'])
        df.at[idx, 'total_cost'] = total_cost
        opt_burden, opt_spacing, opt_explosive = optimize_blast(row)
        df.at[idx, 'optimized_burden'] = opt_burden
        df.at[idx, 'optimized_spacing'] = opt_spacing
        df.at[idx, 'optimized_explosive'] = opt_explosive
        print(f"Blast {row['blastcode']}: Predicted {frag_in_range:.2f}% vs Actual {row['frag_in_range']:.2f}%")

    return df


# Visualization functions
def plot_fragmentation_curve(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x50'], df['predicted_frag_in_range'], c='blue', alpha=0.5, label='Predicted')
    plt.scatter(df['x50'], df['frag_in_range'], c='red', alpha=0.5, label='Actual')
    plt.xlabel('Median Fragment Size (m)')
    plt.ylabel('% In-Range')
    plt.title('Fragmentation Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('fragmentation_curve.png')
    plt.close()


def plot_cost_vs_fragmentation(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_cost'] / 1000, df['predicted_frag_in_range'], c='green', alpha=0.5)
    plt.xlabel('Total Cost ($K)')
    plt.ylabel('% In-Range')
    plt.title('Cost vs Fragmentation Trade-off')
    plt.grid(True)
    plt.savefig('cost_vs_fragmentation.png')
    plt.close()


def plot_safety_bar(df, regulatory_ppv=10):
    plt.figure(figsize=(12, 6))
    subset = df.head(20)
    bar_width = 0.35
    index = np.arange(len(subset))
    plt.bar(index, subset['ppv'], bar_width, label='Actual PPV', color='teal')
    plt.bar(index + bar_width, subset['predicted_ppv'], bar_width, label='Predicted PPV', color='orange')
    plt.axhline(y=regulatory_ppv, color='red', linestyle='--', label='Regulatory PPV')
    plt.xlabel('Blast Code')
    plt.ylabel('PPV (mm/s)')
    plt.title('Safety Compliance (PPV)')
    plt.xticks(index + bar_width / 2, subset['blastcode'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('safety_bar.png')
    plt.close()


# Main execution
if __name__ == '__main__':
    file_path = 'your_cleaned_file.csv'
    df = process_blasts(file_path)
    print("Blast Optimization Summary")
    print(f"Average Fragmentation (% in range): {df['predicted_frag_in_range'].mean():.2f}%")
    print(f"Average Total Cost: ${df['total_cost'].mean() / 1000:.2f}K")
    print(f"Safety Violations (PPV > 10 mm/s): {len(df[df['ppv'] > 10])}")
    high_frag = df[df['frag_in_range'] > 90]
    if not high_frag.empty:
        best_blast = high_frag.iloc[0]
        print(f"Interesting Fact: Blast {best_blast['blastcode']} achieved {best_blast['frag_in_range']}% "
              f"in-range with burden={best_blast['burden']}m, spacing={best_blast['spacing']}m")
    df[['blastcode', 'burden', 'optimized_burden', 'spacing', 'optimized_spacing',
        'total_explosive_kg', 'optimized_explosive', 'total_cost']].to_csv('optimized_parameters.csv', index=False)
    plot_fragmentation_curve(df)
    plot_cost_vs_fragmentation(df)
    plot_safety_bar(df)
    print("Visualizations saved as PNG files.")
    print("Optimized parameters saved to 'optimized_parameters.csv'.")