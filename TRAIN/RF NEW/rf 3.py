import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from deap import base, creator, tools, algorithms
import warnings
import random

warnings.filterwarnings('ignore')

# Load dataset
try:
    data = pd.read_csv('cleaned_blast_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Ensure 'cleaned_blast_data.csv' is in the working directory.")

# Define features and targets
features = [
    'burden', 'holedia', 'spacing', 'hole_depth', 'sremming_length', 'bench_height',
    'hole_angle', 'total_rows', 'hole_blasted', 'column_charge_density',
    'avg_column_charge_length', 'avg_col_weight', 'total_explosive_kg', 'rock_density'
]
targets = ['frag_in_range', 'frag_over_size']

# Feature engineering
data['burden_spacing'] = data['burden'] * data['spacing']
data['depth_to_bench'] = data['hole_depth'] / data['bench_height']
data['explosive_per_hole'] = data['total_explosive_kg'] / data['hole_blasted']
features.extend(['burden_spacing', 'depth_to_bench', 'explosive_per_hole'])

# Handle missing or invalid values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets)
data = data[data['production_ton_therotical'] > 0]

# Prepare data
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipelines
pipelines = {}
preprocessor = ColumnTransformer([('num', StandardScaler(), features)])
for target in targets:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    pipelines[target] = pipeline

# Hyperparameter tuning
param_grid = {'rf__n_estimators': [100], 'rf__max_depth': [10]}
results = {}
for target in targets:
    grid_search = GridSearchCV(pipelines[target], param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train[target])
    pipelines[target] = grid_search.best_estimator_

    # Predictions
    y_test_pred = pipelines[target].predict(X_test)

    # Metrics
    test_r2 = r2_score(y_test[target], y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[target], y_test_pred))

    results[target] = {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'y_test': y_test[target],
        'y_pred': y_test_pred
    }

    # Save model
    joblib.dump(pipelines[target], f'rf_model_{target}.joblib')

# Genetic Algorithm Optimization
param_ranges = {
    'burden': (X['burden'].min(), X['burden'].max()),
    'spacing': (X['spacing'].min(), X['spacing'].max()),
    'total_explosive_kg': (X['total_explosive_kg'].min(), X['total_explosive_kg'].max())
}

# Define fitness and individual
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize frag_in_range, minimize frag_over_size
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float",
                 lambda: [random.uniform(param_ranges[key][0], param_ranges[key][1]) for key in param_ranges])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Evaluation function
def evaluate_individual(individual):
    input_data = X.iloc[0:1].copy()  # Use a DataFrame slice
    input_data['burden'] = individual[0]
    input_data['spacing'] = individual[1]
    input_data['total_explosive_kg'] = individual[2]
    input_data['burden_spacing'] = individual[0] * individual[1]
    input_data['explosive_per_hole'] = individual[2] / input_data['hole_blasted']

    frag_in_range = pipelines['frag_in_range'].predict(input_data)[0]
    frag_over_size = pipelines['frag_over_size'].predict(input_data)[0]

    return frag_in_range, frag_over_size


toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Run GA
random.seed(42)
population = toolbox.population(n=30)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)
best_individual = tools.selBest(population, k=1)[0]

# Predict with optimized parameters
optimized_params = {key: best_individual[i] for i, key in enumerate(param_ranges)}
input_data = X.iloc[0:1].copy()
for key, value in optimized_params.items():
    input_data[key] = value
input_data['burden_spacing'] = optimized_params['burden'] * optimized_params['spacing']
input_data['explosive_per_hole'] = optimized_params['total_explosive_kg'] / input_data['hole_blasted']
optimized_predictions = {target: pipelines[target].predict(input_data)[0] for target in targets}

# Visualizations
plt.figure(figsize=(10, 8))

# Scatter Plots
for i, target in enumerate(targets, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=results[target]['y_test'], y=results[target]['y_pred'], alpha=0.6)
    plt.plot([results[target]['y_test'].min(), results[target]['y_test'].max()],
             [results[target]['y_test'].min(), results[target]['y_test'].max()], 'r--')
    plt.xlabel(f'Actual {target} (%)')
    plt.ylabel(f'Predicted {target} (%)')
    plt.title(f'{target}: R²={results[target]["test_r2"]:.2f}')
    plt.text(0.05, 0.95, f'R²={results[target]["test_r2"]:.2f}', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Feature Importance
plt.subplot(2, 2, 3)
feature_importance = pd.Series(
    pipelines['frag_in_range'].named_steps['rf'].feature_importances_, index=features
).sort_values(ascending=False)[:8]
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title(f'Feature Importance (R²={results["frag_in_range"]["test_r2"]:.2f})')
plt.xlabel('Importance')

plt.tight_layout()
plt.savefig('blast_optimization_plots.png')
plt.close()

# Print summary
print("Model Metrics:")
for target in targets:
    print(f"{target} - Test: R²={results[target]['test_r2']:.2f}, RMSE={results[target]['test_rmse']:.2f}")
print("\nOptimized Parameters:")
for key, value in optimized_params.items():
    print(f"{key}: {value:.2f}")
print("\nPredicted Outcomes with Optimized Parameters:")
for target in targets:
    print(f"{target}: {optimized_predictions[target]:.1f}%")
print("\nModels saved as 'rf_model_<target>.joblib'")
print("Plots saved as 'blast_optimization_plots.png'")
print(
    "\nInspired by: Afolabi et al., 'Optimization of Blasting Parameters Using Regression Models and Genetic Algorithm: A Statistical Approach,' Mathematical Geosciences, 2024. https://doi.org/10.1007/s11004-024-10174-1")