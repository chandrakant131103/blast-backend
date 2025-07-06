import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # <-- force non-interactive backend


# Load your uploaded CSV file
data = pd.read_csv('your_cleaned_file.csv')

# Correct columns based on your file
columns_needed = ['hole_depth', 'spacing', 'burden', 'total_explosive_kg', 'ppv', 'air_blast']
for col in columns_needed:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=columns_needed)

# Assume distance is constant (if you have it, replace here)
distance = 100  # in meters

# Calculate scaled variable W/D
data['W_D'] = data['total_explosive_kg'] / distance

# --------------------------
# Fit PPV model: PPV = k*(W/D)^alpha
def ppv_model(W_D, k, alpha):
    return k * (W_D ** alpha)

popt_ppv, _ = curve_fit(ppv_model, data['W_D'], data['ppv'], p0=[1, 1])
k_ppv, alpha_ppv = popt_ppv
print(f"PPV Model: PPV = {k_ppv:.3f} * (W/D)^{alpha_ppv:.3f}")

# --------------------------
# Fit AOP model: AOP = C*(W/D)^beta
def aop_model(W_D, C, beta):
    return C * (W_D ** beta)

popt_aop, _ = curve_fit(aop_model, data['W_D'], data['air_blast'], p0=[1, 1])
C_aop, beta_aop = popt_aop
print(f"AOP Model: AOP = {C_aop:.3f} * (W/D)^{beta_aop:.3f}")

# --------------------------
# Optimization function

# Define limits
PPV_LIMIT = 10  # mm/s
AOP_LIMIT = 130  # dB

# Cost function (simple example: minimize total explosive weight)
def objective(x):
    spacing, burden, depth, explosive_weight = x
    return explosive_weight  # minimize explosive consumption

# Constraints: vibration limits
def ppv_constraint(x):
    spacing, burden, depth, explosive_weight = x
    W_D = explosive_weight / distance
    ppv = k_ppv * (W_D ** alpha_ppv)
    return PPV_LIMIT - ppv

def aop_constraint(x):
    spacing, burden, depth, explosive_weight = x
    W_D = explosive_weight / distance
    aop = C_aop * (W_D ** beta_aop)
    return AOP_LIMIT - aop

# Design constraints: spacing, burden, depth, weight bounds
bounds = [(2, 5), (2, 5), (3, 15), (10, 100)]  # Adjust based on site condition

# Initial guess
x0 = [3, 3, 5, 50]

# Define constraints in dictionary form for optimizer
constraints = [
    {'type': 'ineq', 'fun': ppv_constraint},
    {'type': 'ineq', 'fun': aop_constraint}
]

result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    spacing_opt, burden_opt, depth_opt, weight_opt = result.x
    print("\nOptimal Design Parameters:")
    print(f"Spacing: {spacing_opt:.2f} m")
    print(f"Burden: {burden_opt:.2f} m")
    print(f"Depth: {depth_opt:.2f} m")
    print(f"Explosive Weight: {weight_opt:.2f} kg")
else:
    print("Optimization failed.")

# --------------------------
# Plot fitted models

# PPV Plot
plt.scatter(data['W_D'], data['ppv'], label='PPV Data')
W_D_range = np.linspace(min(data['W_D']), max(data['W_D']), 100)
plt.plot(W_D_range, ppv_model(W_D_range, *popt_ppv), color='red', label='PPV Fit')
plt.xlabel('W/D')
plt.ylabel('PPV (mm/s)')
plt.legend()
plt.title("PPV Regression")
plt.savefig("blast_model_plot2.png")


# AOP Plot
plt.scatter(data['W_D'], data['air_blast'], label='AOP Data')
plt.plot(W_D_range, aop_model(W_D_range, *popt_aop), color='green', label='AOP Fit')
plt.xlabel('W/D')
plt.ylabel('AOP (dB)')
plt.legend()
plt.title("AOP Regression")
plt.savefig("blast_model_plot3.png")
