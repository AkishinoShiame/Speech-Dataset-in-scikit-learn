import numpy as np

# X stands for Data-set
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y stands for Label
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, Y)
# GaussianNB(priors=None)
print(clf.predict([[-0.8, -1]]))
# [1]
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
# GaussianNB(priors=None)
print(clf_pf.predict([[-0.8, -1]]))
