import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


def load_and_preprocess_data(file_path):
    """Load and preprocess the CSV data."""
    try:
        # Load data
        df = pd.read_csv('your_cleaned_file.csv')

        # Handle missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Encode categorical variables
        categorical_cols = ['rock_name', 'pitname', 'benchname', 'zonename']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Define features and targets
        feature_cols = [
            'burden', 'spacing', 'hole_depth', 'sremming_length', 'holedia',
            'total_explosive_kg', 'rock_density', 'bench_height', 'total_rows',
            'hole_blasted', 'column_charge_density', 'avg_column_charge_length',
            'avg_col_weight', 'blastcode', 'rock_name'
        ]
        target_cols = ['actual_pf_(ton/kg)', 'frag_in_range', 'ppv']

        # Filter valid columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        target_cols = [col for col in target_cols if col in df.columns]

        # Remove rows with zero or negative values in critical columns
        df = df[df['total_explosive_kg'] > 0]
        df = df[df['ton_recover'] > 0]

        # Remove outliers (using IQR)
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

        # Split features and targets
        X = df[feature_cols]
        y = df[target_cols]

        return X, y, feature_cols, target_cols, df

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None, None


def train_xgboost_model(X_train, y_train, X_val, y_val, target_name):
    """Train XGBoost model with Optuna hyperparameter tuning."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'random_state': 42
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred, squared=False)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Train final model with best parameters
    best_params = study.best_params
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    return model, best_params


def evaluate_model(model, X_test, y_test, target_name):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluation for {target_name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return y_pred, rmse, mae, r2


def plot_feature_importance(model, feature_cols, target_name):
    """Plot feature importance."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importance for {target_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target_name}.png')
    plt.close()


def optimize_blast_pattern(models, scaler, feature_cols, target_cols, constraints, df):
    """Recommend optimal blasting parameters."""

    def objective(trial):
        params = {}
        for col in feature_cols:
            if col in ['rock_name', 'blastcode', 'total_rows', 'hole_blasted']:
                params[col] = trial.suggest_int(col, int(df[col].min()), int(df[col].max()))
            else:
                params[col] = trial.suggest_float(col, df[col].min(), df[col].max())

        # Create input array
        input_df = pd.DataFrame([params], columns=feature_cols)
        input_scaled = scaler.transform(input_df)

        # Predict outcomes
        predictions = {}
        for target, model in models.items():
            predictions[target] = model.predict(input_scaled)[0]

        # Objective: Maximize frag_in_range, minimize ppv
        score = predictions['frag_in_range'] - 0.5 * predictions['ppv']

        # Apply constraints
        if predictions['ppv'] > constraints['max_ppv']:
            score -= 1000  # Heavy penalty
        if params['total_explosive_kg'] > constraints['max_explosive']:
            score -= 1000
        if predictions['actual_pf_(ton/kg)'] < constraints['min_pf']:
            score -= 1000

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    input_df = pd.DataFrame([best_params], columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    # Predict outcomes for best parameters
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(input_scaled)[0]

    return best_params, predictions


def main():
    file_path = 'your_cleaned_file.csv'

    # Load and preprocess data
    X, y, feature_cols, target_cols, df = load_and_preprocess_data(file_path)
    if X is None:
        return

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train and evaluate models for each target
    models = {}
    results = {}
    for target in target_cols:
        print(f"\nTraining model for {target}...")
        model, best_params = train_xgboost_model(
            X_train, y_train[target], X_test, y_test[target], target
        )
        models[target] = model
        y_pred, rmse, mae, r2 = evaluate_model(model, X_test, y_test[target], target)
        results[target] = {'y_pred': y_pred, 'rmse': rmse, 'mae': mae, 'r2': r2}

        # Plot feature importance
        plot_feature_importance(model, feature_cols, target)

        # Save model
        model.save_model(f'xgboost_model_{target}.json')

    # Define constraints for optimization
    constraints = {
        'max_ppv': 10.0,  # Max allowable PPV (mm/s)
        'max_explosive': df['total_explosive_kg'].quantile(0.95),  # 95th percentile
        'min_pf': 3.0  # Minimum powder factor
    }

    # Optimize blast pattern
    print("\nOptimizing blast pattern...")
    best_params, predictions = optimize_blast_pattern(
        models, scaler, feature_cols, target_cols, constraints, df
    )

    # Print optimization results
    print("\nOptimal Blast Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value:.2f}")
    print("\nPredicted Outcomes:")
    for target, value in predictions.items():
        print(f"{target}: {value:.2f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        'Parameter': list(best_params.keys()) + list(predictions.keys()),
        'Value': list(best_params.values()) + list(predictions.values()),
        'Type': ['Parameter'] * len(best_params) + ['Prediction'] * len(predictions)
    })
    results_df.to_csv('blast_optimization_results.csv', index=False)

    print("\nResults saved to 'blast_optimization_results.csv'")


if __name__ == "__main__":
    main()