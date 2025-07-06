import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input

# Load data from CSV
df = pd.read_csv('blast_design_data.csv')

# Transpose the DataFrame and set 'Parameter' as index
df = df.set_index('Parameter').T

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Features and target
X = df[['Burden (m)', 'Spacing (m)', 'Hole Depth (m)', 'Sub Drill (m)', 'Stemming (kg)',
        'Powder Factor (kg/m3)', 'PPV (mm/s)', 'Total Volume (m3)',
        'Drilling & Stemming Length (m)', 'Total Explosives (kg)', 'Cost ($/m3)', 'Vibration Control']]
y = df['Stemming (kg)']  # Optimizing for fragmentation

# Convert non-numeric columns to numeric where possible, and handle NaN
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with NaN values
X = X.dropna()
y = y[X.index]

# Check if enough samples are available
if len(X) < 2:
    print("Error: Insufficient data samples after dropping NaN values. Please provide more data.")
else:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build ANN model
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2}")

    # Save the model
    with open('fragmentation_model_adjusted.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

    # Actual vs Predicted graph
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Stemming (kg)')
    plt.ylabel('Predicted Stemming (kg)')
    plt.title('Actual vs Predicted Fragmentation')
    plt.savefig('actual_vs_predicted_adjusted.png')
    plt.close()