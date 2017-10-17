from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle as pkl
import time

X, Y = pkl.load(open("train_data_gray.pkl", "rb"))
test = pkl.load(open("test_data_gray.pkl", "rb"))
test_ans = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print("finished Loading...")

start = time.time()

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
print("Start training ...")
clf.fit(X, Y)
print("Finished training")
print("Total cost", time.time() - start, "sec")


print("Accuracy")
print(clf.score(X, Y)*100.0, "%")
print("Accuracy(Test)")
print(clf.score(test, test_ans)*100.0, "%")

