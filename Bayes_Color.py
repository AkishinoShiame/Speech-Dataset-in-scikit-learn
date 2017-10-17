import numpy as np
import pickle as pkl
from sklearn.naive_bayes import GaussianNB
import time

"""
# X stands for Data-set
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y stands for Label
Y = np.array([1, 1, 1, 2, 2, 2])
"""
X, Y = pkl.load(open("train_data_color.pkl", "rb"))
test = pkl.load(open("test_data_color.pkl", "rb"))
test_ans = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print("finished Loading...")

start = time.time()

clf = GaussianNB()
print("Start training ...")
clf.fit(X, Y)
print("Finished training")
print("Total cost", time.time() - start, "sec")

# GaussianNB(priors=None)
print("Accuracy")
print(clf.score(X, Y)*100.0, "%")
print("Accuracy(Test)")
print(clf.score(test, test_ans)*100.0, "%")

"""
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
# GaussianNB(priors=None)
print(clf_pf.predict([[-0.8, -1]]))
"""