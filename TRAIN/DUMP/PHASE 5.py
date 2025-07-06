import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Assume real-time data is received as JSON or API response
# Example: real-time input simulated
real_time_data = {
    "hole_id": "H123",
    "depth_m": 9.8,
    "rock_hardness": 75,  # MWD sensor
    "gps_location": (23.1234, 82.5678),
    "burden_m": 4.0,
    "spacing_m": 5.0,
    "face_angle": 80,
    "bench_height": 10,
    "previous_ppv": 9.5,
    "previous_airblast": 125,
    "previous_flyrock": 45
}

# Cost coefficients (live prices can be pulled from ERP or inventory systems)
cost_factors = {
    "drilling_cost_per_m3": 10,
    "explosive_cost_per_kg": 200,
    "labor_cost": 5000,
    "accessories_cost": 1000
}

# Real-time prediction models based on updated measurements
def predict_frag(burden, spacing, hardness):
    return 85 - 1.2 * (burden - 4) - 1.0 * (spacing - 4) + 0.05 * hardness

def predict_ppv(burden, spacing, weight):
    return 5 + 0.3 * (spacing - 4) + 0.2 * (weight / 100)

def predict_air(burden, spacing, weight):
    return 120 + 0.5 * (weight / 100)

def predict_flyrock(burden, spacing, weight):
    return 40 + 0.2 * spacing * burden - 0.1 * weight

# Optimization function
def total_cost(x):
    burden, spacing, depth = x
    volume = burden * spacing * depth
    weight = volume * 0.8
    return (
        volume * cost_factors["drilling_cost_per_m3"] +
        weight * cost_factors["explosive_cost_per_kg"] +
        cost_factors["labor_cost"] +
        cost_factors["accessories_cost"]
    )

# Constraints with real-time data injected
def constraints(x):
    burden, spacing, depth = x
    weight = burden * spacing * depth * 0.8
    return [
        predict_frag(burden, spacing, real_time_data['rock_hardness']) - 80,
        10 - predict_ppv(burden, spacing, weight),
        130 - predict_air(burden, spacing, weight),
        50 - predict_flyrock(burden, spacing, weight)
    ]

cons = [{'type': 'ineq', 'fun': lambda x, i=i: constraints(x)[i]} for i in range(4)]
bounds = [(3, 6), (3, 6), (5, 12)]

result = minimize(total_cost, x0=[4, 4, 8], bounds=bounds, constraints=cons)

burden_opt, spacing_opt, depth_opt = result.x
weight_opt = burden_opt * spacing_opt * depth_opt * 0.8

print("----- Real-Time Optimized Design -----")
print(f"Burden: {burden_opt:.2f} m")
print(f"Spacing: {spacing_opt:.2f} m")
print(f"Depth: {depth_opt:.2f} m")

print(f"Explosive Weight: {weight_opt:.2f} kg")
print(f"Total Cost: ₹{total_cost(result.x):,.2f}")
