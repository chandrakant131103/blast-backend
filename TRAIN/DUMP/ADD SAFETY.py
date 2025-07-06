import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

# Load data
data = pd.read_csv('your_cleaned_file.csv')

# Select required columns and drop rows with invalid data
columns_needed = ['burden', 'spacing', 'total_explosive_kg', 'flyrock']
data = data[columns_needed].dropna()

# Filter out rows with zero or negative values
data = data[
    (data['burden'] > 0) &
    (data['spacing'] > 0) &
    (data['total_explosive_kg'] > 0) &
    (data['flyrock'] > 0)
]

# Apply log transformation
data['log_burden'] = np.log(data['burden'])
data['log_spacing'] = np.log(data['spacing'])
data['log_weight'] = np.log(data['total_explosive_kg'])
data['log_flyrock'] = np.log(data['flyrock'])

# Prepare independent and dependent variables
X = data[['log_burden', 'log_spacing', 'log_weight']]
X = sm.add_constant(X)  # add intercept
y = data['log_flyrock']

# Fit linear regression model
model = sm.OLS(y, X).fit()
print(model.summary())

# Extract coefficients
A_log = model.params['const']
b = model.params['log_burden']
c = model.params['log_spacing']
d = model.params['log_weight']
A = np.exp(A_log)

print(f"\nFlyrock Model: Flyrock = {A:.3f} * Burden^{b:.3f} * Spacing^{c:.3f} * Weight^{d:.3f}")

# Define flyrock function (for prediction)
def flyrock_function(burden, spacing, weight):
    return A * (burden ** b) * (spacing ** c) * (weight ** d)

# Define objective for optimization (minimize flyrock while keeping safe parameters)
def objective(x):
    burden, spacing, depth = x
    weight = burden * spacing * depth * 0.8  # Simplified explosive weight calculation
    flyrock_value = flyrock_function(burden, spacing, weight)
    return flyrock_value

# Bounds for burden, spacing, and depth (adjust as per your mine design limits)
bounds = [(3, 8), (3, 8), (3, 15)]

# Run optimization
result = minimize(objective, x0=[4, 4, 6], bounds=bounds)

optimal_burden, optimal_spacing, optimal_depth = result.x
optimal_weight = optimal_burden * optimal_spacing * optimal_depth * 0.8
optimal_flyrock = flyrock_function(optimal_burden, optimal_spacing, optimal_weight)

print("\nOptimal Design Parameters:")
print(f"Burden: {optimal_burden:.2f} m")
print(f"Spacing: {optimal_spacing:.2f} m")
print(f"Depth: {optimal_depth:.2f} m")
print(f"Explosive Weight: {optimal_weight:.2f} kg")
print(f"Predicted Flyrock: {optimal_flyrock:.2f} m")
