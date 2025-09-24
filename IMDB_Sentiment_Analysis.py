#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('IMDB Dataset.csv')


df.head(10)


# In[3]:


# importing the libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split as tts


# In[4]:


# importing the dataset
data = pd.read_csv("IMDB Dataset.csv")


# In[5]:


# top values of the data-set
data.head()


# In[6]:


# shape of the data
data.shape


# In[7]:


# column names 
data.columns


# In[8]:


# count of unique values in the column
data['sentiment'].value_counts()


# In[9]:


# top 10 elements of the dataset
data.head(10)


# In[10]:


# data from the bottom
data.tail(5)


# In[11]:


def clean_text1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

cleaned1=lambda x:clean_text1(x)


# In[12]:


data['review']=pd.DataFrame(data.review.apply(cleaned1))


# In[13]:


data.head()


# In[14]:


# second round of cleaning
def clean_text2(text):
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    return text

cleaned2=lambda x:clean_text2(x)


# In[15]:


data['review']=pd.DataFrame(data.review.apply(cleaned2))
data.head()


# In[16]:


x = data.iloc[0:,0].values
y = data.iloc[0:,1].values


# In[17]:


xtrain,xtest,ytrain,ytest = tts(x,y,test_size = 0.25,random_state = 225)


# In[18]:


tf = TfidfVectorizer()
from sklearn.pipeline import Pipeline


# In[19]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
model=Pipeline([('vectorizer',tf),('classifier',classifier)])

model.fit(xtrain,ytrain)


# In[20]:


ypred=model.predict(xtest)


# In[21]:


# model score
accuracy_score(ypred,ytest)


# In[22]:


# confusion matrix
A=confusion_matrix(ytest,ypred)
print(A)


# In[23]:


# f1 score
recall=A[0][0]/(A[0][0]+A[1][0])
precision=A[0][0]/(A[0][0]+A[0][1])
F1=2*recall*precision/(recall+precision)
print(F1)


# In[ ]:




