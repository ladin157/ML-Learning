from sklearn import datasets
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# let's convert to dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['species'])

# replace the values with class labels
iris.species = np.where(iris.species==0.0, 'setosa', np.where(iris.species==1.0,'versicolor','virginica'))

# let's remove spaces from column name
iris.columns = iris.columns.str.replace(' ','')
iris.describe()

# print(iris['species'].value_counts())
# set the size of the plot
plt.figure(figsize=(15, 8))

iris.hist()
plt.title("Histogram", fontsize=16)
plt.show()