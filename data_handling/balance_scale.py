import pandas as pd
import numpy as np

# read dataset
df = pd.read_csv('balance-scale.data', names=['balance', 'var1', 'var2', 'var3', 'var4'])

# display example observations
print(df.head())

print(df['balance'].value_counts())

df['balance'] = [1 if b == 'B' else 0 for b in df.balance]

print(df['balance'].value_counts())

# the danger of imbalanced classes
print("The danger of imbalanced classes")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# seperate input features (X) and target variable (Y)
y = df.balance
X = df.drop('balance', axis=1)

# train model
clf_0 = LogisticRegression().fit(X, y)

# predict on training set
pred_y_0 = clf_0.predict(X)

# the accuracy
print(accuracy_score(pred_y_0, y))

# should we be excited
print(np.unique(pred_y_0))

# Up-sample minority class
# up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal

# module for resampling
print("Resample dataset")
from sklearn.utils import resample

# we create a new DataFrame with an up-sampled minority class
# steps
# 1. first, we separate observations from each class into different DataFrames
# 2. next, we resample the minority class with replacement
# 3. finally, we combine the up-sampled minority class DataFrame with the original majority class DataFrame

# separate majority and minority classes
df_majority = df[df.balance == 0]
df_minority = df[df.balance == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True, # sample with replacement
 n_samples=576, # to match majority class
 random_state=123 ) # reproducible results

# combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# display new class counts
print(df_upsampled.balance.value_counts())

# training and testing again
# separate input and target
y = df_upsampled.balance
X = df_upsampled.drop('balance', axis=1)

#train model
clf_1 = LogisticRegression().fit(X, y)

# predict on training set
pred_y_1 = clf_1.predict(X)

# is out model still predicting just one class?
print(np.unique(pred_y_1))

# how the accuracy
print(accuracy_score(y, pred_y_1))

print("Down-sampling")
# The process is similar to that of up-sampling
# Steps
# 1. first, we seprate observations from each class into diffenrent DataFrames
# 2. next, we resample the majority class without replacement, setting the number of samples to match that of minority class
# 3. finally, we combine the down-sampled majority class DataFrame with the original minority class DataFrame

# separate
df_majority = df[df.balance==0]
df_minority = df[df.balance==1]

# downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=49,
                                   random_state=123)

# combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# dipaly new class counts
print(df_downsampled.balance.value_counts())

# seperate input features X and target Variable (y)
y = df_downsampled.balance
X = df_downsampled.drop('balance',axis=1)

# train label
clf_2 = LogisticRegression().fit(X, y)

# predict on training set
pred_y_2 = clf_2.predict(X)

# show class
print(np.unique(pred_y_2))

# the accuracy
print(accuracy_score(y_pred=pred_y_2, y_true=y))
