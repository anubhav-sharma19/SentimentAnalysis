#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Using IMDB Opend Data ( Reviews - sentiment analysis)

# ### importing required liabraries 

# In[1]:


import pandas as pd
import numpy as np
#classification report for model evaluation
from sklearn.metrics import classification_report


# ### Import the data 

# In[2]:


sen = pd.read_csv('IMDB Dataset.csv')


# In[3]:


sen.columns


# # # We will us nlp ( nltk) liabrary to eliminate semantics in our reviews
# 1. import `string`
# 2. import `nltk` and `re` (regular expression) 
# 

# In[5]:


import string
import nltk
import re
#corpus have stop words 
from nltk.corpus import stopwords


# We will create a function to clean our reviews data 
# ## it will remove `stopwors` and remove html tags (this is webscrapped data )
# 1.  removing puncuations using string liabrary ..
# 2. removing stopwords in like is the and etc.....

# In[6]:


def clean_text(message):
    clean = re.sub("<br /><br />","",message)
    nopunc = [char for char in clean if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[7]:


from sklearn.model_selection import train_test_split


# In[24]:


X=sen['review']
y=sen['sentiment']
A=X[:1000]
b=y[:1000]

X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.20, random_state=42)


# We will train model on much smaller data so that we can choose estimators 

# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[29]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('BagOfWords', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[30]:


get_ipython().run_cell_magic('time', '', 'pipeline.fit(X_train,y_train)')


# In[31]:


predictions = pipeline.predict(X_test)


# In[32]:


print(classification_report(predictions,y_test))


# In[22]:


# import joblib
# from joblib import dump, load


# In[15]:


# joblib.dump(pipeline,'seniment_analysis.joblib',compress=1) 


# ## we will use SGD classifier now with default parameters

# In[33]:


from sklearn.linear_model import SGDClassifier

pipeline2 = Pipeline([
    ('BagOfWords', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',SGDClassifier())   # train on TF-IDF vectors w/ SGD classifier
])


# In[34]:


get_ipython().run_cell_magic('time', '', 'pipeline2.fit(X_train,y_train)\npredictions2 = pipeline.predict(X_test)\nprint(classification_report(predictions2,y_test))')


# # We will now use gridsearchcv to tune hyperparameters for the entire pipeline using SGD Classifier

# In[52]:


from sklearn.model_selection import GridSearchCV
parameters = {
    'BagOfWords__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'classifier__alpha': (1e-2, 1e-3),
}


# In[53]:


get_ipython().run_cell_magic('time', '', 'gs_clf = GridSearchCV(pipeline2, param_grid=parameters, cv=5)')


# In[54]:


get_ipython().run_cell_magic('time', '', 'gs_clf.fit(X_train,y_train)')


# In[55]:


for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# In[60]:


print(gs_clf.best_estimator_)


# In[61]:


gs_clf.best_score_


# # Here we have observed that MultinomialNB performs good and with respect to the performance of mvb . SGDC is equally better also with hyperparameter tuning the accuracy is declined by 2 , so i will go ahead and choose MNB and will train it with all the data

# In[ ]:




