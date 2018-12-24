# Finalize model with Pickle

# We can use pickle operation to serialize our machine learning algorithms and save the serialized format to a file

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.externals import joblib
# import joblib
import time


file = 'pima-indians-diabetes.data'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(file, names=names)
array = dataframe.values
# print(dataframe)
X = array[:, 0:8]
Y = array[:, 8]
# print(X)
# print(Y)
test_size = 0.33
seed = 7
X_train,X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# Save the model to disk
filename = 'finalized_model.sav'
print('Compare time of dumping: ')
time_start = time.time()
pickle.dump(model, open(filename,'wb'))
time_finish = time.time()
time_total = time_finish-time_start
print(time_total)

time_start = time.time()
joblib.dump(model,'model_dump_joblib.sav')
time_finish = time.time()
time_total = time_finish-time_start
print(time_total)

# Later, load the model from disk to test
print('Compare time of loading: ')
time_start = time.time()
loaded_model = pickle.load(open(filename, 'rb'))
time_finish = time.time()
time_total = time_finish-time_start
print(time_total)
result = loaded_model.score(X_test, Y_test)
print(result)

time_start = time.time()
loaded_model = joblib.load('model_dump_joblib.sav')
time_finish = time.time()
time_total = time_finish-time_start
print(time_total)
result = loaded_model.score(X_test, Y_test)
print(result)