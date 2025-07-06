import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Load and preprocess data
df = pd.read_csv('your_cleaned_file.csv')

# Feature engineering with safety checks
df['B/S_ratio'] = np.where(df['spacing'] > 0, df['burden'] / df['spacing'], np.nan)
df['stemming_ratio'] = np.where(df['hole_depth'] > 0, df['sremming_length'] / df['hole_depth'], np.nan)
df['powder_factor'] = np.where(df['production_ton_therotical'] > 0,
                               df['total_explosive_kg'] / df['production_ton_therotical'],
                               np.nan)

# Handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Filter relevant data
features = [
    'rock_name', 'rock_density', 'burden', 'spacing', 'holedia', 'hole_depth',
    'sremming_length', 'total_explosive_kg', 'B/S_ratio', 'stemming_ratio',
    'powder_factor', 'column_charge_density'
]
target = 'frag_in_range'

# Clean data - remove rows with missing target or extreme values
df_clean = df.dropna(subset=[target] + features).copy()

# Remove extreme values (top/bottom 1%)
for col in ['burden', 'spacing', 'total_explosive_kg', 'powder_factor']:
    q1 = df_clean[col].quantile(0.01)
    q99 = df_clean[col].quantile(0.99)
    df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q99)]

# Split data
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = [f for f in features if f != 'rock_name']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['rock_name']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ))
])

# Train and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Save model
joblib.dump(model, 'blast_optimizer_model.pkl')
print("Model saved as blast_optimizer_model.pkl")

# Create sample input schema
input_sample = X_train.iloc[:1].to_dict(orient='records')[0]
with open('model_input_schema.json', 'w') as f:
    json.dump(input_sample, f, indent=2)

print(f"Training complete. Cleaned dataset size: {len(df_clean)} records")