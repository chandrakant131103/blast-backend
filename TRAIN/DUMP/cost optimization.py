import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load cleaned dataset
data = pd.read_csv('your_cleaned_file.csv')

# Filter and clean data (remove rows with missing or zero values)
cols = ['burden', 'spacing', 'hole_depth', 'total_explosive_kg',
        'drilling_cost', 'total_exp_cost', 'man_power_cost', 'blast_accessoriesdelay_cost',
        'frag_in_range', 'ppv', 'air_blast', 'flyrock']

data = data[cols].dropna()
data = data[(data > 0).all(axis=1)]

# Calculate average cost parameters from real data
avg_drill_cost = data['drilling_cost'].mean()
avg_exp_cost = data['total_exp_cost'].mean()
avg_labor_cost = data['man_power_cost'].mean()
avg_accessories_cost = data['blast_accessoriesdelay_cost'].mean()

# Constraint thresholds (can adjust as per your mine standards)
frag_min = 80    # at least 80% in-range fragmentation
ppv_max = 10     # max allowed PPV (mm/s)
airblast_max = 130  # max airblast (dB)
flyrock_max = 50  # max flyrock (meters)

# Simplified predictive models (replace later with ML models)
def predict_frag(burden, spacing):
    return 85 - 2 * (burden - 4) - 1.5 * (spacing - 4)  # simplified empirical

def predict_ppv(burden, spacing, weight):
    return 5 + 0.2 * (spacing - 4) + 0.3 * (weight / 100)  # simplified

def predict_air(burden, spacing, weight):
    return 120 + 0.5 * (weight / 100)  # simplified

def predict_flyrock(burden, spacing, weight):
    return 40 + 0.1 * spacing * burden - 0.05 * weight  # simplified

# Cost function to minimize
def total_cost(x):
    burden, spacing, depth = x
    drill_cost = burden * spacing * depth * 10  # drilling cost (Rs/volume)
    explosive_weight = burden * spacing * depth * 0.8
    explosive_cost = explosive_weight * 200  # Rs/kg
    labor_cost = 5000
    accessories_cost = 1000
    return drill_cost + explosive_cost + labor_cost + accessories_cost

# Constraints
def constraints(x):
    burden, spacing, depth = x
    weight = burden * spacing * depth * 0.8
    return [
        predict_frag(burden, spacing) - frag_min,            # fragmentation constraint
        ppv_max - predict_ppv(burden, spacing, weight),      # PPV constraint
        airblast_max - predict_air(burden, spacing, weight), # air blast constraint
        flyrock_max - predict_flyrock(burden, spacing, weight)  # flyrock constraint
    ]

# Constraint wrapper for optimizer
cons = [{'type': 'ineq', 'fun': lambda x, i=i: constraints(x)[i]} for i in range(4)]

# Bounds for optimization
bounds = [(3, 8), (3, 8), (3, 15)]

# Run optimization
result = minimize(total_cost, x0=[4, 4, 6], bounds=bounds, constraints=cons)

burden_opt, spacing_opt, depth_opt = result.x
weight_opt = burden_opt * spacing_opt * depth_opt * 0.8

# Output results
print("Optimal Design Parameters:")
print(f"Burden: {burden_opt:.2f} m")
print(f"Spacing: {spacing_opt:.2f} m")
print(f"Depth: {depth_opt:.2f} m")
print(f"Explosive Weight: {weight_opt:.2f} kg")
print(f"Total Cost: ₹{total_cost(result.x):,.2f}")
print(f"Fragmentation: {predict_frag(burden_opt, spacing_opt):.2f}%")
print(f"PPV: {predict_ppv(burden_opt, spacing_opt, weight_opt):.2f} mm/s")
print(f"Air Blast: {predict_air(burden_opt, spacing_opt, weight_opt):.2f} dB")
print(f"Flyrock: {predict_flyrock(burden_opt, spacing_opt, weight_opt):.2f} m")
