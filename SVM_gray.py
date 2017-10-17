import numpy as np
import pickle as pkl
from sklearn import svm
import time

X, Y = pkl.load(open("train_data_gray.pkl", "rb"))
test = pkl.load(open("test_data_gray.pkl", "rb"))
test_ans = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0])

print("finished Loading...")

start = time.time()

"""
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True,
 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
  max_iter=-1, decision_function_shape=’ovr’, random_state=None)
    
"""




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
print(clf.predict(test))
print("Accuracy", clf.score(test, test_ans))
print("Finished !")
print("Time Cost", time.time() - pred_start, "sec")
"""


"""
C : float, optional (default=1.0)
 Penalty parameter C of the error term.

kernel : string, optional (default=’rbf’)
 Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
  ‘precomputed’ or a callable. If none is given, ‘rbf’  will be used. If a callable is given it is used to 
  pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

degree : int, optional (default=3)
 Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

gamma : float, optional (default=’auto’)
 Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.

coef0 : float, optional (default=0.0)
 Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

probability : boolean, optional (default=False)
 Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.


shrinking : boolean, optional (default=True)
 Whether to use the shrinking heuristic.

tol : float, optional (default=1e-3)
 Tolerance for stopping criterion.

cache_size : float, optional
 Specify the size of the kernel cache (in MB).

class_weight : {dict, ‘balanced’}, optional
 Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. 
 The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in
  the input data as n_samples / (n_classes * np.bincount(y))

verbose : bool, default: False
 Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if 
 enabled, may not work properly in a multithreaded context.

max_iter : int, optional (default=-1)
 Hard limit on iterations within solver, or -1 for no limit.

decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
 Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers,
 or the original one-vs-one (‘ovo’) decision function of libsvm which has
 shape (n_samples, n_classes * (n_classes - 1) / 2).

  Changed in version 0.19: decision_function_shape is ‘ovr’ by default.
   New in version 0.17: decision_function_shape=’ovr’ is recommended.
  Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.

random_state : int, RandomState instance or None, optional (default=None)
 The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed 
 used by the random number generator; If RandomState instance, random_state is the random number generator; If None, 
 the random number generator is the RandomState instance used by np.random.

"""

