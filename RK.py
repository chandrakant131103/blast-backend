import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit([[1, 2], [3, 4]], [0, 1])  # Example training

joblib.dump(model, "rf_model_ppv.joblib")
