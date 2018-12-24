# Scikit-learn's Pipeline class is designed as a manageable way to apply a series of data transformations followed by the application of an estimator. In fact, that's really all it is:
#
# Pipeline of transforms with a final estimator.
#
# That's it. Ultimately, this simple tool is useful for:
#
# Convenience in creating a coherent and easy-to-understand workflow
# Enforcing workflow implementation and the desired order of step applications
# Reproducibility
# Value in persistence of entire pipeline objects (goes to reproducibility and convenience)
#
# So let's have a quick look at Pipelines. Specifically, here is what we will do.
#
# Build 3 pipelines, each with a different estimator (classification algorithm), using default hyperparameters:
#
# Logisitic Regression
# Support Vector Machine
# Decision Tree
# To demonstrate pipeline transforms, will perform:
#
# feature scaling
# dimensionality reduction, using PCA to project data onto 2 dimensional space
# We will then end with fitting to our final estimators.
#
# Afterward, and almost completely unrelated, in order to make this a little more like a full-fledged workflow (it still isn't, but closer), we will:
#
# Followup with scoring test data
# Compare pipeline model accuracies
# Identify the "best" model, meaning that which has the highest accuracy on our test data
# Persist (save to file) the entire pipeline of the "best" model

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree

# load and split the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# construct some pipelines
pipe_lr = Pipeline(
    [('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', svm.SVC(random_state=42))])

pipe_dt = Pipeline(
    [('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', tree.DecisionTreeClassifier(random_state=42))])

# list of pipelines for ease of iteration
pipelines = [pipe_lr, pipe_svm, pipe_dt]

# dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Support Vector Machine', 2: 'Decision Tree'}

# fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

# compare accuracies
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X_test, y_test)))

# identity the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''

for idx, val in enumerate(pipelines):
    score = val.score(X_test, y_test)
    if score > best_acc:
        best_acc = score
        best_pipe = val
        best_clf = idx

print('Classifier with best accuracy: %s' % pipe_dict[best_clf])

# save pipeline to file
joblib.dump(best_pipe, 'best_pipeline.pkl', compress=1)
print('Save %s pipeline to file.' % pipe_dict[best_clf])

