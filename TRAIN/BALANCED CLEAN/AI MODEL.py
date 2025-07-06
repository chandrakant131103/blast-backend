# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
# import matplotlib
#
# matplotlib.use('Agg')  # Non-interactive backend
# import matplotlib.pyplot as plt
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # Load and preprocess data
# def load_data(file_path):
#     df = pd.read_csv(file_path, parse_dates=['blastdate'])
#     df = df.dropna(subset=['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'frag_in_range', 'frag_over_size',
#                            'frag_under_size', 'total_drill_mtr'])
#     df = df[df['burden'] > 0]
#     df = df[df['spacing'] > 0]
#     df = df[df['hole_depth'] > 0]
#     df = df[df['total_explosive_kg'] > 0]
#     numeric_cols = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'rock_density',
#                     'frag_in_range', 'frag_over_size', 'frag_under_size', 'ppv', 'flyrock',
#                     'total_drill_mtr', 'total_exp_cost', 'drilling_cost']
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
#     # Convert cm to meters if needed
#     if df['burden'].mean() > 100:
#         df['burden'] /= 100
#         df['spacing'] /= 100
#         df['hole_depth'] /= 100
#     return df
#
#
# # Kuz-Ram model for x50
# def kuz_ram_x50(burden, spacing, hole_depth, explosive_weight, rock_type):
#     A = 1.0 if 'COAL' in rock_type.upper() else 0.8
#     x50 = A * (burden * spacing) ** 0.8 * hole_depth ** 0.1 * explosive_weight ** -0.3
#     return max(0.01, min(x50, 1.0))  # Clamp to 0.01–1.0m
#
#
# # Rosin-Rammler distribution
# def rosin_rammler(x, x50, k=2.2):
#     return (1 - np.exp(-((x / x50) ** k))) * 100  # % passing
#
#
# # Estimate fragmentation metrics
# def estimate_frag_distribution(x50, target_range=(0.05, 0.5)):
#     size_min, size_max = target_range
#     p_min = rosin_rammler(size_min, x50)
#     p_max = rosin_rammler(size_max, x50)
#     frag_in_range = p_max - p_min
#     frag_over_size = 100 - p_max
#     frag_under_size = p_min
#     return frag_in_range, frag_over_size, frag_under_size
#
#
# # Vibration prediction
# def predict_ppv(charge_per_delay, distance=100, k=114, n=1.6):
#     scaled_distance = distance / (charge_per_delay ** (1 / 3))
#     ppv = k * (scaled_distance ** -n)
#     return ppv
#
#
# # Cost calculation
# def calculate_costs(total_drill_mtr, explosive_weight, burden, spacing, orig_burden, orig_spacing, drill_cost_per_m=500,
#                     exp_cost_per_kg=100):
#     drill_mtr = total_drill_mtr * (burden / orig_burden) * (spacing / orig_spacing)
#     drilling_cost = drill_mtr * drill_cost_per_m
#     blasting_cost = explosive_weight * exp_cost_per_kg
#     return drilling_cost + blasting_cost
#
#
# # Optimization function
# def optimize_blast(row, target_frag=80, max_oversize=5, regulatory_ppv=10, drill_cost_per_m=500, exp_cost_per_kg=100):
#     def objective(params):
#         burden, spacing, explosive_weight = params
#         total_cost = calculate_costs(row['total_drill_mtr'], explosive_weight, burden, spacing, row['burden'],
#                                      row['spacing'], drill_cost_per_m, exp_cost_per_kg)
#         return total_cost
#
#     def frag_constraint(params):
#         burden, spacing, explosive_weight = params
#         x50 = kuz_ram_x50(burden, spacing, row['hole_depth'], explosive_weight, row['rock_name'])
#         frag_in_range, _, _ = estimate_frag_distribution(x50)
#         return frag_in_range - target_frag
#
#     def oversize_constraint(params):
#         burden, spacing, explosive_weight = params
#         x50 = kuz_ram_x50(burden, spacing, row['hole_depth'], explosive_weight, row['rock_name'])
#         _, frag_over_size, _ = estimate_frag_distribution(x50)
#         return max_oversize - frag_over_size
#
#     def ppv_constraint(params):
#         _, _, explosive_weight = params
#         charge_per_delay = explosive_weight / (row['total_rows'] or 1)
#         ppv = predict_ppv(charge_per_delay)
#         return regulatory_ppv - ppv
#
#     constraints = [
#         {'type': 'ineq', 'fun': frag_constraint},
#         {'type': 'ineq', 'fun': oversize_constraint},
#         {'type': 'ineq', 'fun': ppv_constraint}
#     ]
#     bounds = [(row['burden'] * 0.7, row['burden'] * 1.3),
#               (row['spacing'] * 0.7, row['spacing'] * 1.3),
#               (row['total_explosive_kg'] * 0.7, row['total_explosive_kg'] * 1.3)]
#     initial_guess = [row['burden'], row['spacing'], row['total_explosive_kg']]
#     result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds, method='SLSQP')
#     return result.x if result.success else initial_guess
#
#
# # Construct cumulative distribution curves
# def get_distribution_curves(df, size_range=np.logspace(-2, 0, 100)):  # 10mm to 1000mm
#     predicted_curves = []
#     actual_curves = []
#
#     for _, row in df.iterrows():
#         # Predicted curve
#         x50 = kuz_ram_x50(row['burden'], row['spacing'], row['hole_depth'], row['total_explosive_kg'], row['rock_name'])
#         predicted = [rosin_rammler(size, x50) for size in size_range]
#         predicted_curves.append(predicted)
#
#         # Actual curve
#         actual = []
#         for size in size_range:
#             if size < 0.05:
#                 p = row['frag_under_size'] * (size / 0.05)
#             elif size <= 0.5:
#                 p = row['frag_under_size'] + row['frag_in_range'] * ((size - 0.05) / (0.5 - 0.05))
#             else:
#                 p = row['frag_under_size'] + row['frag_in_range'] + row['frag_over_size'] * ((size - 0.5) / (1.0 - 0.5))
#             actual.append(min(p, 100))
#         actual_curves.append(actual)
#
#     predicted_avg = np.mean(predicted_curves, axis=0)
#     actual_avg = np.mean(actual_curves, axis=0)
#     return predicted_avg, actual_avg, size_range
#
#
# # Process data
# def process_blasts(file_path):
#     df = load_data(file_path)
#
#     df['predicted_x50'] = 0.0
#     df['predicted_frag_in_range'] = 0.0
#     df['predicted_oversize'] = 0.0
#     df['predicted_under_size'] = 0.0
#     df['total_cost'] = 0.0
#     df['optimized_burden'] = 0.0
#     df['optimized_spacing'] = 0.0
#     df['optimized_explosive'] = 0.0
#     df['optimized_cost'] = 0.0
#
#     for idx, row in df.iterrows():
#         # Kuz-Ram predictions
#         x50 = kuz_ram_x50(row['burden'], row['spacing'], row['hole_depth'], row['total_explosive_kg'], row['rock_name'])
#         frag_in_range, frag_over_size, frag_under_size = estimate_frag_distribution(x50)
#         df.at[idx, 'predicted_x50'] = x50
#         df.at[idx, 'predicted_frag_in_range'] = frag_in_range
#         df.at[idx, 'predicted_oversize'] = frag_over_size
#         df.at[idx, 'predicted_under_size'] = frag_under_size
#
#         # Costs
#         df.at[idx, 'total_cost'] = calculate_costs(row['total_drill_mtr'], row['total_explosive_kg'], row['burden'],
#                                                    row['spacing'], row['burden'], row['spacing'])
#
#         # Optimization
#         opt_burden, opt_spacing, opt_explosive = optimize_blast(row)
#         df.at[idx, 'optimized_burden'] = opt_burden
#         df.at[idx, 'optimized_spacing'] = opt_spacing
#         df.at[idx, 'optimized_explosive'] = opt_explosive
#         df.at[idx, 'optimized_cost'] = calculate_costs(row['total_drill_mtr'], opt_explosive, opt_burden, opt_spacing,
#                                                        row['burden'], row['spacing'])
#
#         print(
#             f"Blast {row['blastcode']}: Predicted In-Range {frag_in_range:.2f}% vs Actual {row['frag_in_range']:.2f}%, "
#             f"Oversize {frag_over_size:.2f}% vs Actual {row['frag_over_size']:.2f}%, "
#             f"Cost ${df.at[idx, 'total_cost'] / 1000:.2f}K vs Optimized ${df.at[idx, 'optimized_cost'] / 1000:.2f}K")
#
#     return df
#
#
# # Plot Kuz-Ram vs Actual fragmentation curve
# def plot_kuzram_vs_actual(predicted_avg, actual_avg, size_range):
#     plt.figure(figsize=(10, 6))
#     plt.plot(size_range * 1000, predicted_avg, label='Kuz-Ram Predicted', color='blue')
#     plt.plot(size_range * 1000, actual_avg, label='Actual', color='red', linestyle='--')
#     plt.xscale('log')
#     plt.xlabel('Fragment Size (mm)')
#     plt.ylabel('Cumulative % Passing')
#     plt.title('Kuz-Ram vs Actual Fragmentation Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('kuzram_vs_actual_curve.png')
#     plt.close()
#
#
# # Plot Cost Savings vs Actual Cost
# def plot_cost_savings(df):
#     plt.figure(figsize=(10, 6))
#     cost_savings = (df['total_cost'] - df['optimized_cost']) / 1000  # $K
#     plt.scatter(df['total_cost'] / 1000, cost_savings, c='green', alpha=0.5)
#     plt.axhline(y=0, color='black', linestyle='--')
#     plt.xlabel('Actual Cost ($K)')
#     plt.ylabel('Cost Savings ($K)')
#     plt.title('Cost Savings vs Actual Cost')
#     plt.grid(True)
#     plt.savefig('cost_savings_vs_actual.png')
#     plt.close()
#
#
# # Plot Oversize Comparison
# def plot_oversize_comparison(df):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(df['predicted_oversize'], df['frag_over_size'], c='purple', alpha=0.5)
#     plt.plot([0, 100], [0, 100], color='black', linestyle='--')  # 1:1 line
#     plt.xlabel('Predicted Oversize (%)')
#     plt.ylabel('Actual Oversize (%)')
#     plt.title('Predicted vs Actual Oversize')
#     plt.grid(True)
#     plt.savefig('oversize_comparison.png')
#     plt.close()
#
#
# # Main execution
# if __name__ == '__main__':
#     file_path = 'your_cleaned_file.csv'
#     df = process_blasts(file_path)
#
#     # Get distribution curves
#     predicted_avg, actual_avg, size_range = get_distribution_curves(df)
#
#     # Plot curves
#     plot_kuzram_vs_actual(predicted_avg, actual_avg, size_range)
#     plot_cost_savings(df)
#     plot_oversize_comparison(df)
#
#     # Summary stats
#     mae_in_range = np.mean(np.abs(df['predicted_frag_in_range'] - df['frag_in_range']))
#     mae_oversize = np.mean(np.abs(df['predicted_oversize'] - df['frag_over_size']))
#     print("Kuz-Ram Model Summary")
#     print(f"Average Predicted Fragmentation (% in range): {df['predicted_frag_in_range'].mean():.2f}%")
#     print(f"Average Actual Fragmentation (% in range): {df['frag_in_range'].mean():.2f}%")
#     print(f"Average Predicted Oversize: {df['predicted_oversize'].mean():.2f}%")
#     print(f"Average Actual Oversize: {df['frag_over_size'].mean():.2f}%")
#     print(f"Average Original Cost: ${df['total_cost'].mean() / 1000:.2f}K")
#     print(f"Average Optimized Cost: ${df['optimized_cost'].mean() / 1000:.2f}K")
#     print(f"Average Cost Savings: ${(df['total_cost'] - df['optimized_cost']).mean() / 1000:.2f}K per blast")
#     print(f"Mean Absolute Error (In-Range): {mae_in_range:.2f}%")
#     print(f"Mean Absolute Error (Oversize): {mae_oversize:.2f}%")
#
#     # Reference Blast 2648
#     blast_2648 = df[df['blastcode'] == '2648']
#     if not blast_2648.empty:
#         row = blast_2648.iloc[0]
#         print(
#             f"Blast 2648: Predicted In-Range {row['predicted_frag_in_range']:.2f}% vs Actual {row['frag_in_range']:.2f}%, "
#             f"Oversize {row['predicted_oversize']:.2f}% vs Actual {row['frag_over_size']:.2f}%, "
#             f"Cost ${row['total_cost'] / 1000:.2f}K vs Optimized ${row['optimized_cost'] / 1000:.2f}K, "
#             f"Burden={row['burden']:.2f}m, Spacing={row['spacing']:.2f}m")
#
#     # Save results
#     df[['blastcode', 'burden', 'spacing', 'total_explosive_kg', 'predicted_x50',
#         'predicted_frag_in_range', 'frag_in_range', 'predicted_oversize', 'frag_over_size',
#         'predicted_under_size', 'frag_under_size', 'total_cost', 'optimized_burden',
#         'optimized_spacing', 'optimized_explosive', 'optimized_cost']].to_csv('kuzram_optimized_predictions.csv',
#                                                                               index=False)
#     print("Plots saved as 'kuzram_vs_actual_curve.png', 'cost_savings_vs_actual.png', 'oversize_comparison.png'")
#     print("Predictions and optimized parameters saved to 'kuzram_optimized_predictions.csv'")


import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['blastdate'])
    df = df.dropna(subset=['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'frag_in_range', 'frag_over_size',
                           'frag_under_size'])
    df = df[df['burden'] > 0]
    df = df[df['spacing'] > 0]
    df = df[df['hole_depth'] > 0]
    df = df[df['total_explosive_kg'] > 0]
    numeric_cols = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg', 'rock_density',
                    'frag_in_range', 'frag_over_size', 'frag_under_size']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Convert cm to meters if needed
    if df['burden'].mean() > 100:
        df['burden'] /= 100
        df['spacing'] /= 100
        df['hole_depth'] /= 100
    return df


# Kuz-Ram model for x50
def kuz_ram_x50(burden, spacing, hole_depth, explosive_weight, rock_type):
    A = 1.0 if 'COAL' in rock_type.upper() else 0.8
    x50 = A * (burden * spacing) ** 0.8 * hole_depth ** 0.1 * explosive_weight ** -0.3
    return max(0.01, min(x50, 1.0))  # Clamp to 0.01–1.0m


# Rosin-Rammler distribution
def rosin_rammler(x, x50, k=2.2):
    return (1 - np.exp(-((x / x50) ** k))) * 100  # % passing


# Estimate fragmentation metrics
def estimate_frag_distribution(x50, target_range=(0.05, 0.5)):
    size_min, size_max = target_range
    p_min = rosin_rammler(size_min, x50)
    p_max = rosin_rammler(size_max, x50)
    frag_in_range = p_max - p_min
    frag_over_size = 100 - p_max
    frag_under_size = p_min
    return frag_in_range, frag_over_size, frag_under_size


# Construct cumulative distribution curves
def get_distribution_curves(df, size_range=np.logspace(-2, 0, 100)):  # 10mm to 1000mm
    predicted_curves = []
    actual_curves = []

    for _, row in df.iterrows():
        # Predicted curve (Kuz-Ram)
        x50 = kuz_ram_x50(row['burden'], row['spacing'], row['hole_depth'], row['total_explosive_kg'], row['rock_name'])
        predicted = [rosin_rammler(size, x50) for size in size_range]
        predicted_curves.append(predicted)

        # Actual curve (approximated)
        actual = []
        for size in size_range:
            if size < 0.05:
                p = row['frag_under_size'] * (size / 0.05)  # Linear interpolation
            elif size <= 0.5:
                p = row['frag_under_size'] + row['frag_in_range'] * ((size - 0.05) / (0.5 - 0.05))
            else:
                p = row['frag_under_size'] + row['frag_in_range'] + row['frag_over_size'] * ((size - 0.5) / (1.0 - 0.5))
            actual.append(min(p, 100))  # Cap at 100%
        actual_curves.append(actual)

    # Average curves
    predicted_avg = np.mean(predicted_curves, axis=0)
    actual_avg = np.mean(actual_curves, axis=0)
    return predicted_avg, actual_avg, size_range


# Process data
def process_blasts(file_path):
    df = load_data(file_path)

    df['predicted_x50'] = 0.0
    df['predicted_frag_in_range'] = 0.0
    df['predicted_oversize'] = 0.0
    df['predicted_under_size'] = 0.0

    for idx, row in df.iterrows():
        x50 = kuz_ram_x50(row['burden'], row['spacing'], row['hole_depth'], row['total_explosive_kg'], row['rock_name'])
        frag_in_range, frag_over_size, frag_under_size = estimate_frag_distribution(x50)
        df.at[idx, 'predicted_x50'] = x50
        df.at[idx, 'predicted_frag_in_range'] = frag_in_range
        df.at[idx, 'predicted_oversize'] = frag_over_size
        df.at[idx, 'predicted_under_size'] = frag_under_size
        print(
            f"Blast {row['blastcode']}: Predicted In-Range {frag_in_range:.2f}% vs Actual {row['frag_in_range']:.2f}%, "
            f"Oversize {frag_over_size:.2f}% vs Actual {row['frag_over_size']:.2f}%")

    return df


# Plot Kuz-Ram vs Actual curve
def plot_kuzram_vs_actual(predicted_avg, actual_avg, size_range):
    plt.figure(figsize=(10, 6))
    plt.plot(size_range * 1000, predicted_avg, label='Kuz-Ram Predicted', color='blue')
    plt.plot(size_range * 1000, actual_avg, label='Actual', color='red', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Fragment Size (mm)')
    plt.ylabel('Cumulative % Passing')
    plt.title('Kuz-Ram vs Actual Fragmentation Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('kuzram_vs_actual_curve.png')
    plt.close()


# Main execution
if __name__ == '__main__':
    file_path = 'your_cleaned_file.csv'
    df = process_blasts(file_path)

    # Get distribution curves
    predicted_avg, actual_avg, size_range = get_distribution_curves(df)

    # Plot curves
    plot_kuzram_vs_actual(predicted_avg, actual_avg, size_range)

    # Summary stats
    mae_in_range = np.mean(np.abs(df['predicted_frag_in_range'] - df['frag_in_range']))
    print("Kuz-Ram Model Summary")
    print(f"Average Predicted Fragmentation (% in range): {df['predicted_frag_in_range'].mean():.2f}%")
    print(f"Average Actual Fragmentation (% in range): {df['frag_in_range'].mean():.2f}%")
    print(f"Average Predicted Oversize: {df['predicted_oversize'].mean():.2f}%")
    print(f"Average Actual Oversize: {df['frag_over_size'].mean():.2f}%")
    print(f"Mean Absolute Error (In-Range): {mae_in_range:.2f}%")

    # Reference Blast 2648
    blast_2648 = df[df['blastcode'] == '2648']
    if not blast_2648.empty:
        row = blast_2648.iloc[0]
        print(
            f"Blast 2648: Predicted In-Range {row['predicted_frag_in_range']:.2f}% vs Actual {row['frag_in_range']:.2f}%, "
            f"Oversize {row['predicted_oversize']:.2f}% vs Actual {row['frag_over_size']:.2f}%, "
            f"Burden={row['burden']:.2f}m, Spacing={row['spacing']:.2f}m")

    # Save results
    df[['blastcode', 'burden', 'spacing', 'total_explosive_kg', 'predicted_x50',
        'predicted_frag_in_range', 'frag_in_range', 'predicted_oversize', 'frag_over_size',
        'predicted_under_size', 'frag_under_size']].to_csv('kuzram_predictions.csv', index=False)
    print("Plot saved as 'kuzram_vs_actual_curve.png'")
    print("Predictions saved to 'kuzram_predictions.csv'")