#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing Datasets

# In[2]:


# READ DATA FROM YEAR 16-17
data_1 = pd.read_excel('C:\\Users\\Yogesh\\OneDrive\\Desktop\\YOGESH\\HR Analytics\\staff utlz latest 16-17_masked.xlsx',header=1)


# In[3]:


data_1.head()


# In[4]:


data_1.shape


# In[5]:


# READ DATA FROM YEAR 17-18
data_2 = pd.read_excel('C:\\Users\\Yogesh\\OneDrive\\Desktop\\YOGESH\\HR Analytics\\staff utlz latest 17-18_masked.xlsx',header=1)


# In[6]:


data_2.head()


# In[7]:


data_2.shape


# In[8]:


# CONCATINATE BOTH DATASETS AND DROP THE DUPLICATES
data = pd.concat([data_1,data_2]).reset_index(drop=True)


# In[9]:


data.shape


# In[10]:


data.head()


# In[11]:


df = data.drop_duplicates(['Employee Name','Employee No'],keep='last')


# In[12]:


df.shape


# In[13]:


df.head()


# In[14]:


df.info()


# In[15]:


df.columns


# ## Cleaning Data

# In[16]:


# DROPING 'MONTHLY WISE HOURS RELATED' COLUMNS AS WE HAVE TOTAL OF THAT FEATURES IN THE END
df.drop(df.iloc[:,11:107],axis=1,inplace=True)


# In[17]:


df.head()


# In[18]:


df.shape


# In[19]:


df.dtypes


# In[20]:


df.isna().sum()


# In[21]:


# VISUALIZING NULL VALUES
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[22]:


# DROPPING COLUMNS
# Decided to drop 'TERMINATION DATE' feature as it carries many null values & does not add much value to the dataset
# (will try ONE-HOT ENCODING later for name columns)
df = df.drop(['Employee No','Employee Name','Termination Date','Supervisor name'],axis=1)


# In[23]:


df.head()


# In[24]:


# Sum of Unique values in each column
{column: len(df[column].unique()) for column in df.columns}


# ## Handling Categorical Features

# In[25]:


# Sum of Unique values in only Categorical columns

{column: len(df[column].unique()) for column in df.select_dtypes('object').columns}


# In[26]:


# converted utilization column from object dtype to float data type

df['Utilization%.12'] = pd.to_numeric(df['Utilization%.12'],errors='coerce')


# In[27]:


# Converting 'People Group' into numerical by Binary Encoding

df['People Group']=df['People Group'].replace({'Client Service Staff':0,'Support Staff':1})


# In[28]:


df.head()


# In[29]:


# Current Status can be Binary Encoded as Active or Not
#'secondment','sabbatical','new joiner' all words mean that the Employee still works for the company

df['Current Status']=df['Current Status'].replace({'Active':1,'Secondment':1,'Sabbatical':1,'New Joiner':1,'Resigned':0})


# In[30]:


df.head()


# In[31]:


# what are the object columns remaining??

{column: len(df[column].unique()) for column in df.select_dtypes('object').columns}


# In[32]:


df['Employee Position'].unique()

# ORDINAL TYPE categorical data which can be ranked
#LEVEL 1 being basic & the rank goes higher as the level increases


# In[33]:


df['Profit Center'].unique()     # ORDINAL TYPE categorical data


# In[34]:


# Binary encoding
# Label encoding for Ordinal type category
# One Hot encoding for Nominal type data where we create dummy variables, used when there are very high number of categories in a feature, top 10 frequently occuring categories are considered
# Count/Frequency encoding for Nominal type data where total count value is assigned to respective unique categories 


# In[35]:


# FOR 'EMPLOYEE POSITION' i will manually rank categories

df['Employee Position']=df['Employee Position'].replace({'Level 1':1,'Level A1':1,'Level 2':2,'Level A2':2,'Level 3':3,
                                                         'Level A3':3,'Level 4':4,'Level 5':5,'Level 6':6,'Level 7':7,'Level 8':8,
                                                         'Level 10':10,'-':0
                                                        })


# In[36]:


# FOR 'PROFIT CENTER' column i will use LabelEncoder

from sklearn import preprocessing
 
label_encoder = preprocessing.LabelEncoder()
 
df['Profit Center']= label_encoder.fit_transform(df['Profit Center'])
 
df['Profit Center'].unique()


# In[69]:


df.head(10)


# In[38]:


# STILL TWO CATEGORICAL FEATURES REMAINING


# In[39]:


# WE CAN APPLY ONE-HOT ENCODING. BUT FROM TWO COLUMNS IT CREATS VERY LARGE NUMBER OF COLUMNS WHICH CAN LEAD TO
# 'CURSE OF DIMENSIONALITY' WHICH MEANS - AS THE NO. OF COLUMNS INCREASES, ACCURACY DECREASES. SO

#a = df.iloc[:,2:5]
#b = a.drop('People Group',axis=1)
#b
#pd.get_dummies(b).shape


# In[40]:


# WILL APPLY COUNT/FREQUENCY ENCODING FOR 'EMPLOYEE LOCATION' AND 'EMPLOYEE CATEGORY'


# In[41]:


df['Employee Location'].value_counts().to_dict()


# In[42]:


df_freq_map = df['Employee Location'].value_counts().to_dict()


# In[43]:


df['Employee Location'] = df['Employee Location'].map(df_freq_map)


# In[44]:


df['Employee Location'].head()


# In[45]:


df_freq1_map = df['Employee Category'].value_counts().to_dict()
df['Employee Category'] = df['Employee Category'].map(df_freq1_map)
df['Employee Category'].head()


# In[46]:


df.head()


# In[47]:


# converting join date column into separate numerical columns

df['year'] = df['Join Date'].dt.year
df['month'] = df['Join Date'].dt.month
df['day'] = df['Join Date'].dt.day
df.head()


# In[48]:


df.drop('Join Date',axis=1,inplace=True)


# In[49]:


df.head()


# In[50]:


# Forgot to reset index
df.reset_index(drop=True,inplace=True)


# In[51]:


df.isna().sum()


# In[52]:


# Three NAN values in Utilization column replaced with '0'
df['Utilization%.12'] = df['Utilization%.12'].fillna(0)


# In[53]:


df.head()


# In[54]:


#df.corr()
# VISUALIZE CORRELATION
#plt.figure(figsize=(16,8))
#sns.heatmap(df.corr(),annot=True,fmt='.0%')


# In[55]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn',fmt='.0%')


# In[57]:


# split df into x & y

y = df['Current Status']
x = df.drop('Current Status',axis=1)

# IMPORT REQUIRED LIBRARIES
# and perform train_test_split

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,shuffle=True,random_state=1)


# In[58]:


# TRAINING THE MODEL
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[59]:


predictions = logmodel.predict(x_test)


# In[60]:


predictions


# In[61]:


# check how are the predictions. Seems to be a Normal Distribution which is not skewed.
sns.distplot(y_test-predictions)


# In[62]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,predictions)


# In[63]:


matrix


# In[64]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy *100)


# In[65]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[66]:




models= {
         'K-Nearest Neighbors':KNeighborsClassifier(),
         'Decision Tree':DecisionTreeClassifier(),
         'Support Vector Machine (Linear Kernel)':LinearSVC(),
         'Support Vector Machine(RBF Kernel)':SVC(),
         'Random Forest':RandomForestClassifier(),
         'Gradient Boosting':GradientBoostingClassifier()
        }

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name +  '--'  +'trained')


# In[67]:


#RESULTS

for name,model in models.items():
    print(name + ': {:.2f}%'.format (model.score(x_test,y_test)*100))


# In[ ]:





# In[ ]:




