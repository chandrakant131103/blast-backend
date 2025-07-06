from flask import Flask, request, jsonify
import joblib
from scipy.optimize import differential_evolution
import numpy as np

# Load model and scaler
model = joblib.load("model_frag_in_range.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# Dynamic bounds from dataset
param_bounds = {
    'burden': (0.0, 16.97),
    'spacing': (0.0, 405.0),
    'sremming_length': (0.0, 19.0),
    'avg_col_weight': (0.0, 1019.52)
}

# Order of features
feature_order = list(param_bounds.keys())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [data.get(feat, 0) for feat in feature_order]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return jsonify({"predicted_frag_in_range": prediction})


@app.route("/optimize", methods=["POST"])
def optimize():
    def objective(x):
        features_scaled = scaler.transform([x])
        prediction = model.predict(features_scaled)[0]
        return -prediction  # maximize frag_in_range

    bounds = list(param_bounds.values())
    result = differential_evolution(objective, bounds, seed=42)

    optimal_input = dict(zip(feature_order, result.x))
    optimal_frag = -result.fun

    return jsonify({
        "optimal_input": optimal_input,
        "predicted_frag_in_range": optimal_frag
    })


if __name__ == "__main__":
    app.run(debug=True)
