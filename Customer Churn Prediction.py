import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report, confusion_matrix

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Churn_Modelling.csv')

numeric_columns = df.select_dtypes(include='number')
dft = df[numeric_columns.columns]

print(dft['Exited'].value_counts())

y = dft['Exited']
X = dft.drop(columns=['Exited'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

Exiting = df[df.Exited == 1]
NotExiting = df[df.Exited == 0]
print("Exiting shape:", Exiting.shape)
print("Not Exiting shape:", NotExiting.shape)
score = accuracy_score(y_test, y_pred)
print('Accuracy:', score)

print('Classification Report:',classification_report(y_pred,y_test))

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)
f1score=f1_score(y_test, y_pred)
print('fq score:',f1score)