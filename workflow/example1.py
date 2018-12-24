# create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# load data
url = os.path.join(os.getcwd(), 'pima-indians-diabetes.data.csv')
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names, sep=',')
# print(dataframe)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# create a feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

# create a pipeline
estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('lda', LinearDiscriminantAnalysis()))
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))

model = Pipeline(estimators)

# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
