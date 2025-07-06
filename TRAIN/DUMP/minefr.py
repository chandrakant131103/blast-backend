# Phase 1: Full Rock Fragmentation Optimization System

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Integer

# 1️⃣ Load and Preprocess Data
data = pd.read_csv('your_cleaned_file.csv')

# Features and target
features = [
    'rock_name', 'rock_density', 'burden', 'spacing', 'hole_depth',
    'sremming_length', 'bench_height', 'face_length', 'hole_angle',
    'total_rows', 'hole_blasted', 'total_explosive_kg',
    'flyrock', 'air_blast', 'ppv'
]
target = 'frag_in_range'

# Clean data: drop rows without target
data_model = data.dropna(subset=[target])
X = data_model[features].copy()
y = data_model[target]

# Encode categorical rock_name
le = LabelEncoder()
X['rock_name'] = le.fit_transform(X['rock_name'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")


# 2️⃣ Optimization Module
# Define optimization function
def optimize(params):
    burden, spacing, hole_depth, sremming_length, total_explosive_kg = params

    # Use default values for non-optimized features
    default = X_train.mean()

    # Build input row
    input_data = {
        'rock_name': default['rock_name'],  # default rock type or you can select specific one
        'rock_density': default['rock_density'],
        'burden': burden,
        'spacing': spacing,
        'hole_depth': hole_depth,
        'sremming_length': sremming_length,
        'bench_height': default['bench_height'],
        'face_length': default['face_length'],
        'hole_angle': default['hole_angle'],
        'total_rows': default['total_rows'],
        'hole_blasted': default['hole_blasted'],
        'total_explosive_kg': total_explosive_kg,
        'flyrock': default['flyrock'],
        'air_blast': default['air_blast'],
        'ppv': default['ppv']
    }

    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    # Objective: minimize distance to target fragmentation
    target_frag = target_fragmentation
    return abs(pred - target_frag)


# Define parameter bounds
space = [
    Real(1, 8, name='burden'),
    Real(1, 8, name='spacing'),
    Real(2, 15, name='hole_depth'),
    Real(1, 10, name='sremming_length'),
    Real(50, 500, name='total_explosive_kg')
]

# 3️⃣ User Input Interface
rock_input = input("\nEnter Rock Type: ").upper()

# Ensure valid rock type
if rock_input not in data_model['rock_name'].unique():
    print("Error: Rock type not found!")
else:
    # Set rock type
    default = data_model[data_model['rock_name'] == rock_input].select_dtypes(include='number').mean()

    encoded_rock = le.transform([rock_input])[0]

    # Get target fragmentation input
    target_fragmentation = float(input("Enter Target Fragmentation Size (Frag in Range %): "))

    # Run Bayesian Optimization
    res = gp_minimize(optimize, space, n_calls=30, random_state=42)

    # Show optimal blast design
    print("\n✅ Optimal Blast Parameters:")
    print(f"Burden: {res.x[0]:.2f} m")
    print(f"Spacing: {res.x[1]:.2f} m")
    print(f"Hole Depth: {res.x[2]:.2f} m")
    print(f"Sremming Length: {res.x[3]:.2f} m")
    print(f"Total Explosive (kg): {res.x[4]:.2f} kg")

    input_data = {
        'rock_name': encoded_rock,
        'rock_density': default['rock_density'],
        'burden': res.x[0],
        'spacing': res.x[1],
        'hole_depth': res.x[2],
        'sremming_length': res.x[3],
        'bench_height': default['bench_height'],
        'face_length': default['face_length'],
        'hole_angle': default['hole_angle'],
        'total_rows': default['total_rows'],
        'hole_blasted': default['hole_blasted'],
        'total_explosive_kg': res.x[4],
        'flyrock': default['flyrock'],
        'air_blast': default['air_blast'],
        'ppv': default['ppv']
    }
    input_df = pd.DataFrame([input_data])
    final_frag = model.predict(input_df)[0]
    print(f"Predicted Fragmentation (Frag in Range): {final_frag:.2f}%")
