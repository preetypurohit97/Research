
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
import gzip
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import numpy as np


# In[2]:


data_path = Path(os.getcwd())/'Desktop'/'data'/'hindi2vec'
assert data_path.exists()
for pathroute in os.walk(str(data_path)):
    next_path = pathroute[1]
    for stop in next_path:
        print(stop)


# In[3]:


train_path = data_path/'train'
test_path = data_path/'test'
    


# In[4]:


def read_data(dir_path):
    
    """read data into pandas dataframe"""
    files_list = list(dir_path.iterdir())
    for filename in files_list:
     review = pd.read_csv(str(filename),sep='\t',encoding="utf-8")
     df =  pd.DataFrame(review)
     df.columns = ["label", "text"]
     return df


# In[5]:


train = read_data(train_path)
test = read_data(test_path)


# In[6]:


X_train, y_train = train['text'], train['label']
X_test, y_test = test['text'], test['label']


# In[7]:


""" FINDING NAN DATA AND REMOVING THEM """
nan_rows_train = X_train[X_train.isnull()]
nan_rows_test = X_test[X_test.isnull()]

X_train=X_train.drop(X_train.index[717])
y_train=y_train.drop(y_train.index[717])

X_test=X_train.drop(X_test.index[264])
y_test=y_train.drop(y_test.index[264])


# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LR


# In[9]:


lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)


# In[10]:


lr_predicted = lr_clf.predict(X_test)

lr_acc = sum(lr_predicted == y_test)/len(lr_predicted)
lr_acc


# In[11]:


from sklearn.metrics import accuracy_score


# In[12]:


def imdb_acc(pipeline_clf):
    predictions = pipeline_clf.predict(X_test)
    assert len(y_test) == len(predictions)
    return sum(predictions == y_test)/len(y_test)


# In[13]:


lr_clf = Pipeline([('vect', CountVectorizer(stop_words= frozenset(["."]))), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)
imdb_acc(lr_clf)


# In[14]:


lr_clf = Pipeline([('vect', CountVectorizer(stop_words= frozenset(["."]), ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)
imdb_acc(lr_clf)


# In[15]:


from sklearn.naive_bayes import MultinomialNB as MNB
mnb_clf = Pipeline([('vect', CountVectorizer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
imdb_acc(mnb_clf)


# In[16]:


mnb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
imdb_acc(mnb_clf)


# In[17]:


mnb_clf = Pipeline([('vect', CountVectorizer(stop_words=frozenset(["."]))), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
imdb_acc(mnb_clf)


# In[18]:


from sklearn.tree import DecisionTreeClassifier as DTC
dtc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',DTC())])
dtc_clf.fit(X=X_train, y=y_train)
imdb_acc(dtc_clf)


# In[19]:


from sklearn.ensemble import RandomForestClassifier as RFC
rfc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',RFC())])
rfc_clf.fit(X=X_train, y=y_train)
imdb_acc(rfc_clf)


# In[20]:


from sklearn.ensemble import ExtraTreesClassifier as XTC
xtc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',XTC())])
xtc_clf.fit(X=X_train, y=y_train)
imdb_acc(xtc_clf)


# In[21]:


from sklearn.model_selection import RandomizedSearchCV
param_grid = dict(clf__C=[50, 75, 85, 100], 
                  vect__stop_words=['english', None],
                  vect__ngram_range = [(1, 1), (1, 3)],
                  vect__lowercase = [True, False],
                 )


# In[22]:


random_search = RandomizedSearchCV(lr_clf, param_distributions=param_grid, n_iter=5, scoring='accuracy', n_jobs=-1, cv=3)
random_search.fit(X_train, y_train)
print('Calculated cross-validation accuracy: {random_search.best_score_}')
print(random_search.best_score_)
best_random_clf = random_search.best_estimator_
best_random_clf.fit(X_train, y_train)
imdb_acc(best_random_clf)

