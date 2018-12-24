# khai bao thu vien

import numpy as np
import matplotlib.pyplot as plt
from  sklearn import neighbors, datasets

# Hien thi mot so du lieu mau

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))

X0 = iris_X[iris_y==0,:]
print('\nSamples from class 0:\n', X0[:5,1:])

X1 = iris_X[iris_y==1,:]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y==2,:]
print('\nSamples from class 2:\n', X2[:5,:])

# tach traing va test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print('Training size: %d' %len(y_train))
print('Test size: %d' %len(y_test))

# set voi k=1, tuc la voi moi diem test data, ta chi xet 1 diem training data gan nhat va lay label cua diem do de du doan cho diem test nay

clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
# trong vi dụ này, sử dụng p=2, tức là khoảng cách ở đây được tính là khoảng cách theo norm 2, có thể thay p=1 hoặc một giá trj nào đó.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:")
print("Predicted labels: ", y_test[20:40])
print("Ground truth: ", y_test[20:40])

# PHuong phap danh gia (evaluation method)
# để đánh giá độ chính xác của thuật toán KNN classifier này, chúng ta xem có bao nhiêu điểm trong test data được dự đoán đúng, lấy số lượng này chia cho tổng số lượng trong tập test data sẽ cho ra độ chính xác
# sử dụng hàm accuracy_score để thực hiện công việc này
from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# tăng lên 10 điểm
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy of 10NN with major voting: %.2f %%' %(100*accuracy_score(y_test, y_pred)))

# danh trong so cho cac diem lan can,, su dung weights='distance' trong thu vien sklean
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
# co hai phuong phap danh trong so la 'distance' va 'uniform'
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy of 10NN (customized weights): %.2f %%' %(100*accuracy_score(y_test, y_pred)))



