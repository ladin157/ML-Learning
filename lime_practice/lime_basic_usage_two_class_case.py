import lime
import sklearn
import numpy as np
import sklearn.ensemble
import sklearn.metrics

# fetching data, training a classifier
# fetching the data, we are using a 2-class subset: atheism and christianity
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism','soc.religion.christian']
newsgroup_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroup_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism','christian']

# let's use the tfidf vectorizer, commonly used for text
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroup_train.data)
test_vectors = vectorizer.transform(newsgroup_test.data)


# use random forest for classification. It's usually hard to understand what random forest are doing, especially with many trees
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroup_train.target)

pred = rf.predict(test_vectors)
f1_score = sklearn.metrics.f1_score(newsgroup_test.target, pred, average='binary')
print('f1-score: ',f1_score)

# the problem here is that this classifier may make a high F score.

#Explaining predictions using lime
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)
print(c.predict_proba([newsgroup_test.data[0]]))

# Now we create an explainer object. We pass the class_names as an argument for prettier display.
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

# we then generate an explaination with at most 6 features for an arbitrary document in the test set
idx = 83
exp = explainer.explain_instance(newsgroup_test.data[idx], c.predict_proba, num_features=50)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroup_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroup_test.target[idx]])
print(exp.as_list())
