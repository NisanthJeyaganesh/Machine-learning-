import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Load dataset
df = pd.read_csv('Iris.csv')

# Drop unnecessary column
df.drop('Id', axis=1, inplace=True)

# Encode species labels
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save cleaned dataset
df.to_csv('Iris_Cleaned.csv', index=False)

# Split features and target
X = df[numerical_cols]
y = df['Species_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Export the trained model
joblib.dump(model, 'iris_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to predict species from input values
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load('iris_model.pkl')
    le = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return le.inverse_transform(prediction)[0]

# Example usage
print("Predicted species:", predict_species(5.1, 3.5, 1.4, 0.2))

# Bar chart of species count
plt.figure(figsize=(6, 4))
df['Species'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Species Count in Iris Dataset')
plt.xlabel('Species')
plt.ylabel('Count')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('species_count.png')
plt.show()
