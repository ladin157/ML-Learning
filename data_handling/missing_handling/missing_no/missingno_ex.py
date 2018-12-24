# this piece of blocks the warning messages
import warnings

warnings.filterwarnings('ignore')

# import libsraries and check the versions
import pandas as pd
import sys
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas_profiling

print('Python version: ' + sys.version)
print('Numpy version: ' + np.__version__)
print('Pandas version: ' + pd.__version__)
print('Matplotlib version: ' + matplotlib.__version__)
print('Missingno version: ' + msno.__version__)

# create data with missing values
data = {'name': ['Michael', 'Jessica', 'Sue', 'Jake', 'Amy', 'Tye'],
        'gender': [None, 'F', np.NaN, 'F', np.NaN, 'M'],
        'height': [123, 145, 100, np.NaN, None, 150],
        'weight': [10, np.NaN, 30, np.NaN, None, 20],
        'age': [14, None, 29, np.NaN, 52, 45],
        }
df = pd.DataFrame(data, columns=['name', 'gender', 'height', 'weight', 'age'])
print(df)

# create a numpy array that has a missing value
a = np.array([1, 2, np.nan, 4])
print(a.dtype)

# sum does not work how it is expected
print(np.sum(a))

# use nansum for expected result
print(np.nansum(a))

# detect missing values
# use .info(), isnull(), notnull() for detecting missing values
print(df.info())

# sum of the missing values in each column
print(df.isnull().sum())

# notnull() is opposite of isnull()
print(df.notnull().sum())

# missingno is a greate package to quickly display missing values in a dataset.

msno.matrix(df.sample(6))

msno.bar(df.sample(6))

# pandas_profiling is another package for missing data that gives a high level overview of the dataset as well as detailed information for each column in the dataset including the number of missing values
report = pandas_profiling.ProfileReport(df)
print(report.html)

plt.show()
