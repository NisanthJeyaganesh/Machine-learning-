import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
boston_data = fetch_openml(name='boston', version=1, as_frame=True)
df = boston_data.frame
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Boston Housing Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
medv_corr = correlation_matrix['MEDV'].drop(labels='MEDV')
top_corr_features = medv_corr.abs().sort_values(ascending=False).head(5)
print("Top 5 features most correlated with MEDV:")
print(top_corr_features)
