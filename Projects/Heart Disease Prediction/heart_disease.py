import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/kaggle/input/heart-disease/heart_disease_data.csv')

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training

# Logistic Regression
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

prediction = model.predict(X_test)

output = pd.DataFrame({"Age": X_test.age,"Prediction": prediction , "Target": Y_test})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
