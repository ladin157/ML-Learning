import sklearn
from sklearn import datasets
import sklearn.ensemble
from sklearn.model_selection import train_test_split
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

np.random.seed(1)

# continuous features
# loading data, training a model
iris = datasets.load_iris()
train, test, labels_train, labels_test = train_test_split(iris.data, iris.target, train_size=0.80)

# fit the model
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

acc_score = sklearn.metrics.accuracy_score(labels_test, rf.predict(test))
print(acc_score)

# Create an explainer
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train, feature_names=iris.feature_names,
                                                   class_names=iris.target_names, discretize_continuous=True)

# Explaining an instance
i = 10 # np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=3, top_labels=1)
exp.save_to_file('result.html')
out = exp.as_pyplot_figure()
plt.show()
