import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import product

# Load dataset
df = pd.read_csv('your_cleaned_file.csv')

# Define features and target
features = [
    'burden', 'spacing', 'hole_depth', 'sremming_length',
    'avg_col_weight', 'base_exp_weight', 'total_explosive_kg',
    'ppv', 'drill_factor', 'drilling_cost'
]
target = 'frag_in_range'

X = df[features].fillna(0)
y = df[target].fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter ranges
alpha_range = np.linspace(0.01, 10, 20)
tol_range = np.linspace(0.0001, 0.01, 10)
param_grid = list(product(alpha_range, tol_range))

# Optimization loop
results = []
for alpha, tol in param_grid:
    model = Ridge(alpha=alpha, tol=tol)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    results.append({
        'alpha': alpha,
        'tol': tol,
        'mse': mse
    })

# Best result
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['mse'].idxmin()]
print("Best Parameters:")
print(best)
