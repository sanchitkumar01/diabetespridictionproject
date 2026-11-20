#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('Diabetes_Prediction_Dataset.csv')


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[11]:


data.duplicated().sum()


# In[12]:


for col in ['Gender', 'Smoking_Status', 'Physical_Activity', 'Family_History', 'Diabetes']:
    print(col, data[col].unique())


# In[13]:


data = data.drop('Patient_ID', axis=1)


# In[14]:


data['Diabetes'] = data['Diabetes'].map({'No': 0, 'Yes': 1})
data['Diabetes'].value_counts()


# In[15]:


sns.countplot(x=data['Diabetes'])
plt.show()


# In[16]:


sns.boxplot(data['Glucose'], orient="h")
plt.show()


# In[17]:


sns.boxplot(data['BMI'], orient="h")
plt.show()


# In[18]:


plt.figure(figsize=(8,6))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()


# In[19]:


data = pd.get_dummies(data, columns=['Gender', 'Smoking_Status', 'Physical_Activity', 'Family_History'], drop_first=True, dtype=int)


# In[20]:


data.head()


# In[21]:


x = data.drop('Diabetes', axis=1)
y = data['Diabetes']


# In[22]:


x


# In[23]:


y


# In[25]:


get_ipython().system('pip install scikit-learn xgboost')


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[31]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

x_train_clean = imputer.fit_transform(x_train)
x_test_clean = imputer.transform(x_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train_clean, y_train)

rf = RandomForestClassifier()
rf.fit(x_train_clean, y_train)

gb = GradientBoostingClassifier()
gb.fit(x_train_clean, y_train)

xg = XGBClassifier()
xg.fit(x_train_clean, y_train)


# In[34]:


x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]


# In[36]:


x_test = x_test.dropna()
y_test = y_test.loc[x_test.index]

y_pred1 = lr.predict(x_test)
y_pred2 = rf.predict(x_test)
y_pred3 = gb.predict(x_test)
y_pred4 = xg.predict(x_test)


# In[39]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
x_test_clean = imputer.fit_transform(x_test)

y_pred1 = lr.predict(x_test_clean)
y_pred2 = rf.predict(x_test_clean)
y_pred3 = gb.predict(x_test_clean)
y_pred4 = xg.predict(x_test_clean)


# In[41]:


from sklearn import metrics


# In[42]:


score1 = metrics.accuracy_score(y_test, y_pred1)
score2 = metrics.accuracy_score(y_test, y_pred2)
score3 = metrics.accuracy_score(y_test, y_pred3)
score4 = metrics.accuracy_score(y_test, y_pred4)


# In[43]:


print(score1, score2, score3, score4)


# In[44]:


final_data = pd.DataFrame({'Models': ['LR', 'RF', 'GB', 'XG'], 'Accuracy': [score1, score2, score3, score4]})
final_data


# In[45]:


sns.barplot(x=final_data['Models'], y=final_data['Accuracy'])
plt.show()


# In[46]:


print(metrics.classification_report(y_test, y_pred4))


# In[47]:


sns.heatmap(metrics.confusion_matrix(y_test, y_pred4), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[48]:


xg = XGBClassifier()
xg_final = xg.fit(x, y)


# In[49]:


import joblib


# In[50]:


joblib.dump(xg_final, 'diabetes_predictor')


# In[51]:


model = joblib.load('diabetes_predictor')


# In[52]:


x.columns


# In[53]:


new = {'Age': 45, 'BMI': 28.5, 'Glucose': 140, 'Blood_Pressure': 85,
       'Cholesterol': 200, 'Insulin': 100, 'HbA1c': 6.5,
       'Gender_Male': 1, 'Smoking_Status_Never': 1, 'Smoking_Status_Former': 0,
       'Physical_Activity_Low': 0, 'Physical_Activity_Moderate': 1,
       'Family_History_Yes': 1}

data_new = pd.DataFrame(new, index=[0]).reindex(columns=x.columns, fill_value=0)
data_new


# In[54]:


pred = model.predict(data_new)
print("Diabetes" if pred[0] == 1 else "No Diabetes")


# In[55]:


model.predict_proba(data_new)


# In[56]:


import pickle

with open('diabetes_predictor.pkl', 'wb') as file:
    pickle.dump(xg_final, file)


# In[57]:


import pickle

with open('diabetes_predictor.pkl', 'rb') as file:
    model = pickle.load(file)


# In[58]:


import joblib

joblib.dump(xg_final, 'diabetes_predictor.pkl')


# In[59]:


import joblib

model = joblib.load('diabetes_predictor.pkl')


# In[ ]:




