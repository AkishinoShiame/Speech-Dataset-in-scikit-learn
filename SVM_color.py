import numpy as np
import pickle as pkl
from sklearn import svm
import time

X, Y = pkl.load(open("train_data_color.pkl", "rb"))
test = pkl.load(open("test_data_color.pkl", "rb"))
test_ans = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0])

print("finished Loading...")

start = time.time()

clf = svm.SVC(decision_function_shape='ovo')
print("Start training ...")
clf.fit(X, Y)
print("Finished training")
print("Total cost", time.time() - start, "sec")

print("Accuracy")
print(clf.score(X, Y)*100.0, "%")
# print(clf.predict(test))

print("Accuracy(Test)")
print(clf.score(test, test_ans)*100.0, "%")

"""
pred_start = time.time()
print("Start predicting...")
print("Accuracy", clf.score(X, Y))
print(clf.predict(test))
print("Test Accuracy", clf.score(test, test_ans))
print("Finished !")
print("Time Cost", time.time() - pred_start, "sec")
"""

