import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # <-- force non-interactive backend

# Load cleaned data
file_path = "your_cleaned_file.csv"
df = pd.read_csv(file_path)

# Feature selection
features = ['burden', 'spacing', 'hole_depth', 'column_charge_density', 'total_explosive_kg']
target = 'ton_recover'

# Handle missing values (as double-check)
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df[target] = pd.to_numeric(df[target], errors='coerce')
df.dropna(subset=features + [target], inplace=True)

# Prepare X and y
X = df[features]
y = df[target]

# Split into train-test (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🔧 Model Performance")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Show model coefficients
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\nModel Coefficients:")
print(coefficients)

# Actual vs Predicted plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Ton Recover')
plt.ylabel('Predicted Ton Recover')
plt.title('Actual vs Predicted Ton Recover')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.savefig("blast_model_plot.png")


# Example: Predict for new blast design
sample_blast = pd.DataFrame({
    'burden': [7],
    'spacing': [8],
    'hole_depth': [12],
    'column_charge_density': [0.9],
    'total_explosive_kg': [6000]
})

predicted_ton = model.predict(sample_blast)
print("\nPredicted Ton Recover for new blast design:", predicted_ton[0])
