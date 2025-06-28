import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# loading a dataset
df=pd.read_csv('Iris.csv')

#plotting a histogram for sepal length
plt.figure(figsize=(8, 5))
plt.hist(df['SepalLengthCm'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Sepal Length")
plt.xlabel("SepalLength")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# scatterplot for sepallength vs sepalwidth
plt.figure(figsize=(8, 5))
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], color='green', edgecolors='black', alpha=0.7)
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.grid(True)
plt.show()

#plotting pairplot of a dataset
df = df.drop(columns=["Id"], errors='ignore')
sns.pairplot(df, hue="Species", diag_kind="hist", palette="husl")
plt.show()

#box plot for petal width grouped by species
plt.figure(figsize=(8, 5))
sns.boxplot(x='Species', y='PetalWidthCm', data=df, palette='pastel')
plt.title("Box Plot of Petal Width by Species")
plt.xlabel("Species")
plt.ylabel("Petal Width (cm)")
plt.grid(True)
plt.show()