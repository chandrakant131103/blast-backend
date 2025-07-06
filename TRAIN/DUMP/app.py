from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Load and preprocess dataset
df = pd.read_csv("your_cleaned_file.csv")


# Fix column name mismatch
df.rename(columns={"sremming_length": "stemming_length"}, inplace=True)

# Filter rows where frag_in_range > 0
df = df[df['frag_in_range'] > 0].copy()

# Define features and target
features = ['burden', 'spacing', 'stemming_length', 'avg_col_weight']
target = 'frag_in_range'
cost_target = 'total_exp_cost'

# Train predictive model
X = df[features]
y = df[target]
cost_y = df[cost_target]

frag_model = RandomForestRegressor(n_estimators=100, random_state=42)
frag_model.fit(X, y)

cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
cost_model.fit(X, cost_y)


# Define objective to minimize cost and maximize fragmentation
def combined_objective(params):
    input_df = pd.DataFrame([params], columns=features)
    predicted_frag = frag_model.predict(input_df)[0]
    predicted_cost = cost_model.predict(input_df)[0]

    # Combine objectives: maximize fragmentation, minimize cost
    score = -predicted_frag + 0.01 * predicted_cost
    return score


# Bounds for each parameter (domain knowledge based)
bounds = [
    (2.0, 5.0),  # burden
    (3.0, 6.0),  # spacing
    (1.0, 4.0),  # stemming_length
    (5.0, 25.0)  # avg_col_weight
]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        x0 = [
            float(input_data['burden']),
            float(input_data['spacing']),
            float(input_data['stemming_length']),
            float(input_data['avg_col_weight'])
        ]

        # Run optimization
        result = minimize(combined_objective, x0, bounds=bounds, method='L-BFGS-B')
        optimal_values = dict(zip(features, result.x))

        # Predict frag and cost from optimal parameters
        optimal_df = pd.DataFrame([result.x], columns=features)
        predicted_frag = frag_model.predict(optimal_df)[0]
        predicted_cost = cost_model.predict(optimal_df)[0]

        return jsonify({
            "predicted_frag": predicted_frag,
            "predicted_cost": predicted_cost,
            "optimal_values": optimal_values
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
