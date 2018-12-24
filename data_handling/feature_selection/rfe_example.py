# https://medium.com/@aneesha/recursive-feature-elimination-with-scikit-learn-3a2cbdf23fb7
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets

dataset = datasets.load_iris()

svm = LinearSVC()
# Create the RFE model for the svm classifier and select attributes
rfe = RFE(svm, 3)
rfe = rfe.fit(dataset.data, dataset.target)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)