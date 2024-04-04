# -*- coding: utf-8 -*-
"""notebook98af2a829b

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/notebook98af2a829b-13568294-df6b-4951-8905-ab183ab76251.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240404/auto/storage/goog4_request%26X-Goog-Date%3D20240404T210547Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D96e29f03009d65f6180342eab3ece0e2a785953a55e27560c1cec4ad3d3c8b7fd8ff15551c3820b96ee0d1b2345d1e4ed39be564fe35b0eaf3d8878fb338494b6300da73dd49b7e59f063c87fc114da2dd5482b033b5ba9b0436f1a910371372cddfbf1e1f0c0d15d2da87ba5de0a069f341e6e245d982ec2d9820ce23e905c3406ea79406460ab80ad1f654965dda5e81add575fddfb3488fe5f6d7e064a25ef5f46568d5971ee84742264018e7d30f40c0c6a369b22740e38dffe57d9542bb9789ae16f0250205abff7839902e7d574b9e9541e1a990f9e3fe64dccc01c32b98605f9a551ce96f676c88ea87f5ba156f44b45620dffe42da0e6fbc18047751
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/kaggle/input/heart-disease/heart_disease_data.csv')

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training

# Logistic Regression
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)# training the Random Forest model with Training data
model.fit(X_train, Y_train)

prediction = model.predict(X_test)

output = pd.DataFrame({"Age": X_test.age,"Prediction": prediction , "Target": Y_test})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

