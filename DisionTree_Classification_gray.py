import numpy as np
import pickle as pkl
from sklearn import tree
import time

X, Y = pkl.load(open("train_data_gray.pkl", "rb"))
test = pkl.load(open("test_data_gray.pkl", "rb"))
test_ans = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print("finished Loading...")

start = time.time()

clf = tree.DecisionTreeClassifier()
print("Start training ...")
clf = clf.fit(X, Y)
print("Finished training")
print("Total cost", time.time() - start, "sec")

print("Accuracy")
print(clf.score(X, Y)*100.0, "%")
# print(clf.predict(test))

print("Accuracy(Test)")
print(clf.score(test, test_ans)*100.0, "%")

