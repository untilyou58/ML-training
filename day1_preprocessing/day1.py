#Step 1: Importing the libraries
import numpy as np
import pandas as pd

# Step 2: Importing dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print(X)
print(Y)

# Step 3: Handling the missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print(X)

# Step 4: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print(X)

# Creating a dummy variable
onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(X)
print(Y)

# Step 5: Splitting the datasets into training sets and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

# step 6: Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
print(Y_train)