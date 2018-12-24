# importing pandas
import pandas as pd

# importing training dataset
X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')

# Importing testing data set
X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('Y_test.csv')

print(X_train.head())

# feature scaling
import matplotlib.pyplot as plt

plt.show(X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values].hist(figsize=[11,11]))

# Initializing and Fitting a k-NN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']],Y_train)

# Checking the performance of our model on the testing data set
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome', 'CoapplicantIncome',
                             'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))

print(accuracy)