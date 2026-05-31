# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:54:13 2026

@author: Niranjana
"""
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

heart = pd.read_csv("heart_disease_uci.csv")

print(heart.head())

print(heart.info())

print(heart.isnull().sum())

heart.dropna(inplace=True)

print(heart.isnull().sum())

le = LabelEncoder()

heart['sex'] = le.fit_transform(heart['sex'])
heart['dataset'] = le.fit_transform(heart['dataset'])
heart['cp'] = le.fit_transform(heart['cp'])
heart['fbs'] = le.fit_transform(heart['fbs'])
heart['restecg'] = le.fit_transform(heart['restecg'])
heart['exang'] = le.fit_transform(heart['exang'])
heart['slope'] = le.fit_transform(heart['slope'])
heart['thal'] = le.fit_transform(heart['thal'])

heart['num'] = heart['num'].apply(lambda x: 0 if x == 0 else 1)

print(heart.head())

x = heart[['age',
           'sex',
           'cp',
           'trestbps',
           'chol',
           'thalch',
           'exang',
           'oldpeak',
           'ca',
           'thal']]

y = heart['num']

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

model = RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    max_depth=10,
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
train_accuracy = model.score(x_train, y_train)

print("Training Accuracy:", train_accuracy)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

mlp.fit(x_train, y_train)

plt.plot(mlp.loss_curve_)

plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()

age = int(input("Enter Age: "))
sex = int(input("Enter Sex (1 for Male, 0 for Female): "))
cp = int(input("Enter Chest Pain Type (0-3): "))
bp = float(input("Enter Blood Pressure: "))
chol = float(input("Enter Cholesterol: "))
thalch = float(input("Enter Heart Rate: "))
exang = int(input("Exercise Angina? (1 Yes / 0 No): "))
oldpeak = float(input("Enter Oldpeak Value: "))
ca = int(input("Enter Number of Major Vessels (0-3): "))
thal = int(input("Enter Thal Value (0-2): "))

new_patient = pd.DataFrame([[age,
                             sex,
                             cp,
                             bp,
                             chol,
                             thalch,
                             exang,
                             oldpeak,
                             ca,
                             thal]],

columns=['age',
         'sex',
         'cp',
         'trestbps',
         'chol',
         'thalch',
         'exang',
         'oldpeak',
         'ca',
         'thal'])

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Heart Disease Risk Present")
else:
    print("No Heart Disease Risk")
