import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Load your file
data = pd.read_csv('your_cleaned_file.csv')

# 2️⃣ Feature selection
features = [
    'rock_name', 'rock_density', 'burden', 'spacing', 'hole_depth',
    'sremming_length', 'bench_height', 'face_length', 'hole_angle',
    'total_rows', 'hole_blasted', 'total_explosive_kg',
    'flyrock', 'air_blast', 'ppv'
]
target = 'frag_in_range'

# 3️⃣ Drop missing target values (if any)
data_model = data.dropna(subset=[target])

# 4️⃣ Extract features and target
X = data_model[features].copy()
y = data_model[target]

# 5️⃣ Encode categorical variables
le = LabelEncoder()
X['rock_name'] = le.fit_transform(X['rock_name'])

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Train RandomForest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # manually calculate RMSE (for old sklearn)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# 🔟 Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance_df)

import pickle

# Save the model to a pickle file
with open('random_forest_ppv_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model exported successfully to random_forest_ppv_model.pkl")