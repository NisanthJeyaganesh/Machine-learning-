import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#data loading
df=pd.read_csv('Iris.csv')

#data encoding 
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])
df.drop('Species',axis=1)
print(df)

#splitting dataset into features and target
X = df.drop(['Id','Species', 'Species_encoded'], axis=1)  
y = df['Species_encoded'] 

print(X)
print(y)

#splitting train and test sets
x_train,x_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#training a model
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(x_train, Y_train)

#predicting on a test dataset
y_pred=logreg_model.predict(x_test)

#evaluation
accuracy = accuracy_score(Y_test, y_pred)
target_names = le.classes_
report = classification_report(Y_test, y_pred, target_names=target_names)
conf_matrix = confusion_matrix(Y_test, y_pred)


print(accuracy)
print(report)
print(conf_matrix)

#decision tree classifying
dectree = DecisionTreeClassifier(random_state=42)
dectree.fit(x_train, Y_train)

#y predict
y_pred1 = dectree.predict(x_test)


#evaluation
target_names = le.classes_  
accuracy1 = accuracy_score(Y_test, y_pred1)
report1 = classification_report(Y_test, y_pred1, target_names=target_names)
conf_matrix1 = confusion_matrix(Y_test, y_pred1)


print(accuracy1)
print(report1)
print(conf_matrix1)