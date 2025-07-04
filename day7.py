import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# === Load Boston dataset manually (since load_boston is removed) ===
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Convert the raw format into usable DataFrame
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

df_sklearn = pd.DataFrame(data, columns=column_names)
df_sklearn['MEDV'] = target

print("✅ Sklearn (manual) dataset converted to DataFrame:")
print(df_sklearn.head())

# === Load Kaggle dataset ===
df_kaggle = pd.read_csv("housing.csv")

print("✅ Kaggle dataset loaded:")
print(df_kaggle.head())
