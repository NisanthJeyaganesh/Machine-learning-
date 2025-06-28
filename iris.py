import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#loading dataset
df=pd.read_csv('Iris.csv')
print(df)

#general statistics
print(df.describe())
categorical_cols = df.select_dtypes(include='object').columns
print(df[categorical_cols].describe())

#filtering rows where petallength greater than 1.5
filtered_df = df[df['PetalLengthCm'] > 1.5]
print(filtered_df)

#Encoding species
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])
print(df[['Species', 'Species_encoded']])

#creating petal ratio
df['Petalratio'] = df['PetalLengthCm']/df['PetalWidthCm']
print(df[['PetalLengthCm','PetalWidthCm','Petalratio']])