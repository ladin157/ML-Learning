# When working with classification and/or regression techniques, its always good to have the ability to ‘explain’ what your model is doing. Using Local Interpretable Model-agnostic Explanations (LIME), you now have the ability to quickly provide visual explanations of your model(s).

from sklearn.datasets import load_boston
import sklearn.ensemble
import numpy as np
from sklearn.model_selection import train_test_split
import lime_practice
import lime_practice.lime_tabular

# load boston dataset
boston = load_boston()
# print(boston['DESCR'])

# Use RandomForest as Classifier
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
train, test, labels_train, labels_test = train_test_split(boston.data, boston.target, train_size=0.80)
print(train)
# rf.fit(train, labels_train)

# print('Random Forest MSError', np.mean((rf.predict(test) - labels_test) ** 2))

