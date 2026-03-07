import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("D://pythonMachineLearning//firstMLCode//logistic_regression//Social_Network_Ads.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Features (X):")
print(X.head())
print("Target (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

##independent feature scaling
print("Training Data:")
print(X_train.head())

##dependent feature scaling
print("Testing Data:")
print(X_test.head())

## feature scaling
sc = StandardScaler()
#Best practice: always fit (or fit_transform) on training data only 
# and transform validation/test data to avoid data leakage from test set
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


print("Scaled Training Data:")
print(X_train[:5])
print("Scaled Testing Data:")
print(X_test[:5])

#fitting logistic regression to training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)