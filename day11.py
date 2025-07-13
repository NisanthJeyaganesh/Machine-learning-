import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
boston_data = fetch_openml(name='boston', version=1, as_frame=True)
df = boston_data.frame
numeric_features = df.select_dtypes(include=['float64', 'int64'])
correlations = numeric_features.corr()
medv_correlations = correlations['MEDV'].drop('MEDV')
top_5_features = medv_correlations.abs().sort_values(ascending=False).head(5).index.tolist()
print("Top 5 features most correlated with MEDV:", top_5_features)
X = df[top_5_features]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output results
print("\n=== Linear Regression Evaluation ===")
print(f"Mean Squared Error (MSE):      {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score:                       {r2:.2f}")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted MEDV")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
