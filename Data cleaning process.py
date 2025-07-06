import pandas as pd

# Load the CSV file
df = pd.read_csv('Gevra_1_1_2023_to_12_31_2024 (1)(List Of Blast).csv')

# List of features and targets to consider
features = [
    'Burden', 'HoleDia', 'Spacing', 'Hole Depth', 'Sremming Length', 'Bench Height',
    'Hole Angle', 'Total Rows', 'Hole Blasted', 'Column Charge Density',
    'Avg column Charge Length', 'Avg col weight', 'Total Explosive Kg', 'Rock Density'
]
targets = ['Frag In Range', 'Frag Over Size', 'ppv']

# Combine features and targets for the cleaning process
columns_to_consider = features + targets

# 1. Drop columns that have 90% or more null values
threshold = len(df) * 0.9  # 90% of total rows
df_cleaned = df.dropna(thresh=threshold, axis=1)

# 2. Drop rows that have null values in our target columns
# Only drop rows where all target columns are null
df_cleaned = df_cleaned.dropna(subset=targets, how='all')

# 3. For our features, drop rows where any of the essential features are null
# We'll keep rows where at least some features are present
df_cleaned = df_cleaned.dropna(subset=features, how='all')

# 4. Drop columns that are not in our features/targets list and have no value for our analysis
# First identify which of our selected columns still exist in the dataframe
existing_columns = [col for col in columns_to_consider if col in df_cleaned.columns]
other_columns = [col for col in df_cleaned.columns if col not in columns_to_consider]

# For other columns, drop if they have more than 50% null values
for col in other_columns:
    if df_cleaned[col].isnull().mean() > 0.5:
        df_cleaned.drop(col, axis=1, inplace=True)

# Display information about the cleaned dataframe
print("Original shape:", df.shape)
print("Cleaned shape:", df_cleaned.shape)
print("\nColumns kept in cleaned dataframe:")
print(df_cleaned.columns.tolist())

# Optionally, save the cleaned dataframe to a new CSV file
df_cleaned.to_csv('Mine_Swamp.csv', index=False)