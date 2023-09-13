import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#<---Train Data--->
dtrain = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Copy of fraudTrain.csv')
numeric_columns = dtrain.select_dtypes(include='number')
dtraint = dtrain[numeric_columns.columns]
y = dtraint['is_fraud']
X = dtraint.drop(columns=['is_fraud'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("'\ndata after training\n'")

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)


score = accuracy_score(y_val, y_pred)
print('Accuracy:', score)
print(classification_report(y_pred,y_val))

cm = confusion_matrix(y_val, y_pred)
print('Confusion Matrix:\n', cm)

# <---Test Data--->
dtest = pd.read_csv('//content/drive/MyDrive/Colab Notebooks/fraudTest.csv')
numeric_columns = dtest.select_dtypes(include='number')
dtest = dtest[numeric_columns.columns]
print('\n data after testing\n')
print(dtest['is_fraud'].value_counts())


fraud = dtest[dtest.is_fraud == 1]
legal = dtest[dtest.is_fraud == 0]

print("Fraud data shape:", fraud.shape)
print("Legal data shape:", legal.shape)


y_test = dtest['is_fraud']
X_test = dtest.drop(columns=['is_fraud'])



y_test_pred = model.predict(X_test)


score = accuracy_score(y_test, y_test_pred)
print('Accuracy:', score)

print(classification_report(y_test_pred,y_test))

cm = confusion_matrix(y_test, y_test_pred)
print('Confusion Matrix:\n', cm)


