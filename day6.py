import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# === Manually Load Boston Dataset from CMU ===
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Define column names
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

# Create DataFrame
df_sklearn = pd.DataFrame(data, columns=column_names)
df_sklearn['MEDV'] = target

# Standardize column names
df_sklearn.columns = df_sklearn.columns.str.strip().str.upper()

# === Load from Kaggle CSV ===
df_kaggle = pd.read_csv("housing.csv")  # Make sure this file is in your working directory
df_kaggle.columns = df_kaggle.columns.str.strip().str.upper()

# Optional fix: rename PRICE to MEDV if needed
if "PRICE" in df_kaggle.columns and "MEDV" not in df_kaggle.columns:
    df_kaggle.rename(columns={"PRICE": "MEDV"}, inplace=True)

# === Debug print column names ===
print("📄 Sklearn columns:", sorted(df_sklearn.columns.tolist()))
print("📄 Kaggle columns:", sorted(df_kaggle.columns.tolist()))

# === Compare Columns ===
diff_columns = set(df_kaggle.columns).symmetric_difference(set(df_sklearn.columns))
print("🧾 Column differences:", diff_columns)

# === Compare Shapes ===
shape_diff = {
    "kaggle_shape": df_kaggle.shape,
    "sklearn_shape": df_sklearn.shape
}
print("📐 Shape differences:", shape_diff)

# === Compare Summary Statistics ===
common_cols = list(set(df_kaggle.columns).intersection(set(df_sklearn.columns)))
print("✅ Common columns for comparison:", common_cols)

stat_diff_dict = {}

if not common_cols:
    print("❌ No common columns found between the two datasets. Cannot compute statistics.")
else:
    stat_kaggle = df_kaggle[common_cols].describe().round(2)
    stat_sklearn = df_sklearn[common_cols].describe().round(2)

    stat_diff = (stat_kaggle - stat_sklearn).dropna(how='all')
    print("📊 Summary Statistics Differences:\n", stat_diff)

    # Save stats to dictionary
    stat_diff_dict = stat_diff.to_dict()

# === Save Differences to JSON ===
differences = {
    "column_diff": list(diff_columns),
    "shape_diff": shape_diff,
    "stat_diff": stat_diff_dict
}

with open("boston_dataset_differences.json", "w") as f:
    json.dump(differences, f, indent=4)

print("✅ Differences saved to 'boston_dataset_differences.json'")
