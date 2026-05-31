# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:51:39 2026

@author: Niranjana
"""

import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

heart = pd.read_csv("heart_disease_uci.csv")

heart.dropna(inplace=True)

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

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    max_depth=10,
    random_state=42
)

model.fit(x_train, y_train)

pickle.dump(model, open('heart_model.pkl', 'wb'))

print("Model Saved Successfully")