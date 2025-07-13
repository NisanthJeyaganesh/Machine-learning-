import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load Boston Housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Display basic info
print("=== Dataset Summary ===")
print(df.describe())

print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Missing Values Count ===")
print(df.isnull().sum())

# Boxplots for selected features
selected_features = ['MEDV', 'CRIM']
for feature in selected_features:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f"Boxplot of {feature}")
    plt.xlabel(feature)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Histogram of Target Variable
plt.figure(figsize=(8, 4))
sns.histplot(df['MEDV'], bins=30, kde=True, color='salmon')
plt.title("Distribution of MEDV (Target)")
plt.xlabel("MEDV")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Separate features and target
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Apply scaling
scaler_std = StandardScaler()
scaler_mm = MinMaxScaler()

X_std = pd.DataFrame(scaler_std.fit_transform(X), columns=[col + "_std" for col in X.columns])
X_mm = pd.DataFrame(scaler_mm.fit_transform(X), columns=[col + "_mm" for col in X.columns])

# Combine scaled data with target
df_scaled = pd.concat([y, X_std, X_mm], axis=1)

# Scaled data summaries
print("\n=== Standard Scaled Feature Summary ===")
print(X_std.describe())

print("\n=== MinMax Scaled Feature Summary ===")
print(X_mm.describe())

# KDE plots: original vs scaled for selected features
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# CRIM - Original and Standard Scaled
sns.kdeplot(df['CRIM'], ax=axes[0, 0], color='darkblue')
axes[0, 0].set_title("Original CRIM")
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

sns.kdeplot(X_std['CRIM_std'], ax=axes[0, 1], color='green')
axes[0, 1].set_title("Standard Scaled CRIM")
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# ZN - Original and MinMax Scaled
sns.kdeplot(df['ZN'], ax=axes[1, 0], color='purple')
axes[1, 0].set_title("Original ZN")
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

sns.kdeplot(X_mm['ZN_mm'], ax=axes[1, 1], color='orange')
axes[1, 1].set_title("MinMax Scaled ZN")
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
