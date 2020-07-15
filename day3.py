# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

# print(X)
# print(Y)

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# labelencoder = LabelEncoder()
# X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)

# print(X)
# Avoiding Dummy Variable Trap
X = X[: , 1:]
# print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(np.shape(X_train))
print(np.shape(Y_train))
# Step 3: Predicting the Test set results
plt.scatter(X_train , Y_train, color = 'red')
y_pred = regressor.predict(X_test)
