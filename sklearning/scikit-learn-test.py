
# coding: utf-8

# In[2]:

import sys
sys.path.append('//')


# In[3]:

from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model


# # Pre Processing

# In[4]:

import pandas as pd

get_ipython().magic('time')

df = pd.read_csv('csic raw.csv',sep=',',header=0,low_memory=False)
len(df.columns)



# In[5]:

df.values[137251]


# # Feature Extactor

# In[6]:

class EmptyFitMixin:
    def fit(self, x, y=None):
        return self


# In[7]:

class ItemSelector(BaseEstimator, TransformerMixin, EmptyFitMixin):
    def __init__(self, key):
        self.key = key

    def transform(self, df):
        values = np.array(df[self.key])
        return values.reshape(values.shape[0], 1)


# In[8]:

class TextExtractor(BaseEstimator, TransformerMixin, EmptyFitMixin):
    def __init__(self, text_cols=['index', 'method', 'url', 'protocol', 'userAgent', 'pragma',
       'cacheControl', 'accept', 'acceptEncoding', 'acceptCharset',
       'acceptLanguage', 'host', 'connection', 'contentLength', 'contentType',
       'cookie', 'payload']):
        self.text_cols = text_cols
    
    def transform(self, data):
        def join(items):
            return ' '.join([str(item) for item in items])
            #return ' '.join([str(lxml.html.tostring(item)) for item in items])
        
        texts = data[self.text_cols].apply(join, axis=1)
        return texts


# In[9]:

class LengthExtractor(BaseEstimator, TransformerMixin, EmptyFitMixin):
    def transform(self, data):
        values = np.array([len(d) for d in data])
        return values.reshape(values.shape[0], 1)



# In[10]:

class copy(BaseEstimator,TransformerMixin,EmptyFitMixin):
    def transfrom(self, data):
        values = np.array(col)
        return values.reshape(values.shape[0], 1)


# In[11]:

feature_union_both = FeatureUnion(
    transformer_list=[
        ('index', Pipeline([
            ('index', TextExtractor(text_cols=['index'])),
            ('tfidf', TfidfVectorizer()),
        ])),   
        ('method', Pipeline([
            ('method', TextExtractor(text_cols=['method'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('url', Pipeline([
            ('url', TextExtractor(text_cols=['url'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('protocol', Pipeline([
            ('protocol', TextExtractor(text_cols=['protocol'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('userAgent', Pipeline([
            ('userAgent', TextExtractor(text_cols=['userAgent'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('pragma', Pipeline([
            ('pragma', TextExtractor(text_cols=['pragma'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('cacheControl', Pipeline([
            ('cacheControl', TextExtractor(text_cols=['cacheControl'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('accept', Pipeline([
            ('accept', TextExtractor(text_cols=['accept'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('acceptEncoding', Pipeline([
            ('acceptEncoding', TextExtractor(text_cols=['acceptEncoding'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('acceptCharset', Pipeline([
            ('acceptCharset', TextExtractor(text_cols=['acceptCharset'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('acceptLanguage', Pipeline([
            ('acceptLanguage', TextExtractor(text_cols=['acceptLanguage'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('host', Pipeline([
            ('host', TextExtractor(text_cols=['host'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('connection', Pipeline([
            ('connection', TextExtractor(text_cols=['connection'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('contentLength', Pipeline([
            ('contentLength', TextExtractor(text_cols=['contentLength'])),
            ('tfidf', TfidfVectorizer()),
        ])),
        ('contentType', Pipeline([
            ('contentType', TextExtractor(text_cols=['contentType'])),
            ('tfidf', TfidfVectorizer()),
        ])),
    ],
)


# In[12]:

get_ipython().run_cell_magic('time', '', '\nX_both = feature_union_both.fit_transform(df)')


# # Feature Selection

# In[13]:

get_ipython().run_cell_magic('time', '', 'sel = VarianceThreshold(threshold=(.01))\nX_both_new = sel.fit_transform(X_both)')


# # Training

# In[14]:

get_ipython().run_cell_magic('time', '', "X_both_train, X_both_test, y_both_train, y_both_test = train_test_split(X_both, df['label'], test_size=0.33)")


# In[15]:

X_both_test.nnz


# In[21]:

get_ipython().run_cell_magic('time', '', '# model_randomforest = RandomForestClassifier(n_jobs=10)\n# model_randomforest.fit(X_both_train, y_both_train)\nknn = KNeighborsClassifier(n_jobs=100,n_neighbors=3)\nknn.fit(X_both_train, y_both_train) ')


# In[ ]:

# y_pred_randomforest = model_randomforest.predict(X_both_test)
knn.predict(X_both_test)

# print(classification_report(y_both_test, y_pred_randomforest))


# In[ ]:

# joblib.dump(feature_union_both, '/home/thinh/Documents/feature_union.joblib')
# joblib.dump(model_randomforest, '/home/thinh/Documents/model_randomforest.joblib')


# In[ ]:

# extraction = joblib.load('/home/thinh/Documents/feature_union.joblib')
# model = joblib.load('/home/thinh/Documents/model_randomforest.joblib')



# In[ ]:

d = {'index':[''], 'method':['GET'], 'url':['http://localhost:8080/tienda1/imagenes/3.gif'], 'protocol':['HTTP/1.1'], 'userAgent':['Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)'], 'pragma':['no-cache'],
       'cacheControl':['no-cache'], 'accept':['text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5'], 'acceptEncoding':['x-gzip, x-deflate, gzip, deflate'],
       'acceptCharset':['utf-8, utf-8;q=0.5, *;q=0.5'],
       'acceptLanguage':['en'], 'host':['localhost:8080'], 'connection':['close'], 'contentLength':['null'], 'contentType':['null']}

dp = pd.DataFrame(d)
pd.DataFrame(d)


# In[ ]:

from sklearn.metrics import accuracy_score
x_testing = extraction.transform(dp)
y_testing = model.predict(x_testing)
print(y_testing)


# In[ ]:



