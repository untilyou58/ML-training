# from __future__ import print_function
# import numpy as np 
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# np.random.seed(22)

# means = [[2, 2], [4, 2]]
# cov = [[.3, .2], [.2, .3]]
# N = 10
# X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
# X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
# X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
# y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 

# from cvxopt import matrix, solvers
# # build K
# V = np.concatenate((X0.T, -X1.T), axis = 1)
# K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

# p = matrix(-np.ones((2*N, 1))) # all-one vector 
# # build A, b, G, h 
# G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
# h = matrix(np.zeros((2*N, 1)))
# A = matrix(y) # the equality constrain is actually y^T lambda = 0
# b = matrix(np.zeros((1, 1))) 
# solvers.options['show_progress'] = False
# sol = solvers.qp(K, p, G, h, A, b)

# l = np.array(sol['x'])
# print('lambda = ')
# print(l.T)

# epsilon = 1e-6 # just a small number, greater than 1e-9
# S = np.where(l > epsilon)[0]

# VS = V[:, S]
# XS = X[:, S]
# yS = y[:, S]
# lS = l[S]
# # calculate w and b
# w = VS.dot(lS)
# b = np.mean(yS.T - w.T.dot(XS))

# print('w = ', w.T)
# print('b = ', b)


# With library sklearn.svm import SVC
# from sklearn.svm import SVC

# y1 = y.reshape((2*N,))
# X1 = X.T # each sample is one row
# clf = SVC(kernel = 'linear', C = 1e5) # just a big number 

# clf.fit(X1, y1) 

# w = clf.coef_
# b = clf.intercept_
# print('w = ', w)
# print('b = ', b)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()