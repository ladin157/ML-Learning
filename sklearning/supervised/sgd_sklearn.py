import numpy as np
from sklearn import linear_model

X = np.array([[-1, -1], [-2, -1], [1, 1, ], [2, 1]])
Y = np.array([1, 1, 2, 2])
clf = linear_model.SGDClassifier()
clf.fit(X, Y)
# print(clf)
# SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#        learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
#        n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
#        shuffle=True, tol=None, verbose=0, warm_start=False)

y_pred = clf.predict([[1, 0]])
print(y_pred)