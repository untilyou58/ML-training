# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
# from matplotlib.backends.backend_pdf import PdfPages

# # XOR dataset and targets
# X = np.c_[(0, 0),
#           (1, 1),
#           #---
#           (1, 0),
#           (0, 1)].T
# Y = [0] * 2 + [1] * 2
# # figure number
# fignum = 1

# # fit the model
# for kernel in ('sigmoid', 'poly', 'rbf'):
#     clf = svm.SVC(kernel=kernel, gamma=4, coef0 = 0)
#     clf.fit(X, Y)
#     with PdfPages(kernel + '2.pdf') as pdf:
#         # plot the line, the points, and the nearest vectors to the plane
#         fig, ax = plt.subplots()
#         plt.figure(fignum, figsize=(4, 3))
#         plt.clf()

#         plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#                     facecolors='None')
#         plt.plot(X[:2, 0], X[:2, 1], 'ro', markersize = 8)
#         plt.plot(X[2:, 0], X[2:, 1], 'bs', markersize = 8)

#         plt.axis('tight')
#         x_min, x_max = -2, 3
#         y_min, y_max = -2, 3
        
#         XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#         Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

#         # Put the result into a color plot
#         Z = Z.reshape(XX.shape)
#         plt.figure(fignum, figsize=(4, 3))
#         CS = plt.contourf(XX, YY, np.sign(Z), 200, cmap='jet', alpha = .2)
#         plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                     levels=[-.5, 0, .5])
#         plt.title(kernel, fontsize = 15)
#         plt.xlim(x_min, x_max)
#         plt.ylim(y_min, y_max)

#         plt.xticks(())
#         plt.yticks(())
#         fignum = fignum + 1
#         pdf.savefig()
# plt.show()

import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score # for evaluating results

A = sio.loadmat('myARgender.mat')
X_train = A['Y_train'].T 
X_test = A['Y_test'].T 
N = 700
y_train = A['label_train'].reshape(N)
y_test = A['label_test'].reshape(N)

clf = SVC(kernel='poly', degree = 3, gamma=1, C = 100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))