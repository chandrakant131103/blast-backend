import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

# 1. Load CSV
df = pd.read_csv("your_cleaned_file.csv")

# 2. Define Inputs and Target
features = ['burden', 'spacing', 'sremming_length', 'avg_col_weight']
target = 'frag_in_range'

X = df[features]
y = df[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
print("🔧 Model Performance")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 6. Define Optimization Objective
def objective(params):
    input_df = pd.DataFrame([params], columns=features)
    prediction = model.predict(input_df)[0]
    return -prediction  # Negative to maximize frag_in_range

# 7. Define Parameter Bounds (adjust as per real range if needed)
bounds = [
    (2.0, 5.0),    # burden
    (3.0, 6.0),    # spacing
    (1.0, 4.0),    # sremming_length
    (5.0, 25.0)    # avg_col_weight
]

# 8. Initial Guess
x0 = [3.0, 4.0, 2.0, 12.0]

# 9. Run Optimization
result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

# 10. Output Results
optimal_values = dict(zip(features, result.x))
predicted_frag = -result.fun  # because we minimized the negative

print("\n✅ Optimized Blast Pattern:")
for k, v in optimal_values.items():
    print(f"{k}: {v:.2f}")
print(f"Predicted frag_in_range: {predicted_frag:.2f}")
