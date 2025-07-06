import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Dummy train example — use your actual training data
X = np.random.rand(100, 5)
y = np.random.rand(100)

model1 = RandomForestRegressor().fit(X, y)
model2 = RandomForestRegressor().fit(X, y)
model3 = RandomForestRegressor().fit(X, y)

# Save all models (replacing the existing ones)
joblib.dump(model1, "rf_model_frag_in_range.joblib")
joblib.dump(model2, "rf_model_frag_over_size.joblib")
joblib.dump(model3, "rf_model_ppv.joblib")


print("process done")
