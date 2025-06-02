#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib


# In[2]:


import warnings


# In[3]:


import numpy as np


# In[5]:


import pandas as pd


# In[6]:


from sklearn.svm import SVC


# In[7]:


import plotly.express as px


# In[8]:


import statsmodels.api as sm


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


from scipy.stats import loguniform


# In[11]:


import statsmodels.formula.api as smf


# In[12]:


from sklearn.naive_bayes import GaussianNB


# In[13]:


from sklearn.feature_selection import RFECV


# In[14]:


from tensorflow.keras.optimizers import Adam


# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[16]:


from tensorflow.keras.models import Sequential


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


from tensorflow.keras.layers import Dense, Dropout


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


from sklearn.pipeline import Pipeline


# In[22]:


from sklearn.model_selection import StratifiedKFold


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


from tensorflow.keras.callbacks import EarlyStopping


# In[25]:


from sklearn.model_selection import RandomizedSearchCV


# In[26]:


from sklearn.model_selection import RepeatedStratifiedKFold


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix


# In[28]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[29]:


warnings.filterwarnings('ignore')


# In[30]:


RANDOM_STATE = 42


# In[31]:


pd.set_option('display.max_columns', None)


# In[32]:


data_df = pd.read_csv("churn.csv")


# In[33]:


data_df.columns = data_df.columns.str.strip()


# In[34]:


def dataoveriew(df, message):
    print(f'{message}:')
    print('Number of rows: ', df.shape[0])
    print("Number of features:", df.shape[1])
    print("Data Features:")
    print(df.columns.tolist())
    print("Missing values:", df.isnull().sum().values.sum())
    print("Unique values:")
    print(df.nunique())


# In[35]:


dataoveriew(data_df, 'Overview of the dataset')


# The data fall under two categories:
# - 17 Categorical features:
#     - CustomerID: Customer ID unique for each customer
#     - gender: Whether the customer is a male or a female
#     - SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
#     - Partner: Whether the customer has a partner or not (Yes, No)
#     - Dependent: Whether the customer has dependents or not (Yes, No)
#     - PhoneService: Whether the customer has a phone service or not (Yes, No)
#     - MultipeLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
#     - InternetService: Customer’s internet service provider (DSL, Fiber optic, No)
#     - OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
#     - OnlineBackup: Whether the customer has an online backup or not (Yes, No, No internet service)
#     - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
#     - TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
#     - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
#     - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
#     - Contract: The contract term of the customer (Month-to-month, One year, Two years)
#     - PaperlessBilling: The contract term of the customer (Month-to-month, One year, Two years)
#     - PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
#     
# ***
#     
# - 3 Numerical features:
# 
#     - Tenure: Number of months the customer has stayed with the company 
#     - MonthlyCharges: The amount charged to the customer monthly
#     - TotalCharges: The total amount charged to the customer
#       
# ***
# 
# - Prediction feature:
#     - Churn: Whether the customer churned or not (Yes or No)
#     
#     
# These features can also be sub-divided into:
# 
# - Demographic customer information
# 
#     - gender , SeniorCitizen , Partner , Dependents
# 
# - Services that each customer has signed up for
# 
#     - PhoneService , MultipleLines , InternetService , OnlineSecurity , OnlineBackup , DeviceProtection , TechSupport , StreamingTV , StreamingMovies, 
#     
# - Customer account information
# 
#     - tenure , Contract , PaperlessBilling , PaymentMethod , MonthlyCharges , TotalCharges

# ### Explore Target variable

# In[36]:


target_instance = data_df["Churn"].value_counts().reset_index()


# In[37]:


target_instance.columns = ['Category', 'Count']


# In[38]:


fig = px.pie(
    target_instance,
    values='Count',
    names='Category',
    color='Category',
    color_discrete_sequence=["#FFFF99", "#FFF44F"],  # canary, lemon
    color_discrete_map={"No": "#E30B5C", "Yes": "#50C878"},  # raspberry, emerald
    title='Distribution of Churn'
)


# In[39]:


fig.show()


# We are trying to predict users that left the company in the previous month. It is a binary classification problem with an unbalance target.
# - Churn: No - 73.5%
# - Churn: Yes - 26.5%

# In[40]:


data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'], errors='coerce')
data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())


# In[41]:


data_df['SeniorCitizen'] = data_df['SeniorCitizen'].astype(str).replace({'0': 'No', '1': 'Yes'})


# In[42]:


data_df.drop(["customerID"], axis=1, inplace=True)


# In[43]:


def binary_map(feature):
    return feature.map({'Yes': 1, 'No': 0})


# In[44]:


binary_list = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
data_df[binary_list] = data_df[binary_list].apply(binary_map)
data_df['gender'] = data_df['gender'].map({'Male': 1, 'Female': 0})
data_df['Churn'] = data_df['Churn'].map({'Yes': 1, 'No': 0})


# In[45]:


data_df = pd.get_dummies(data_df, drop_first=True)


# In[46]:


X = data_df.drop('Churn', axis=1)
y = data_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


# In[47]:


cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']


# In[48]:


pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rfecv', RFECV(estimator=LogisticRegression(), cv=StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE), scoring='accuracy')),
    ('model', LogisticRegression(class_weight='balanced'))
])


# In[49]:


pipeline.fit(X_train, y_train)


# In[50]:


y_pred = pipeline.predict(X_test)


# In[51]:


print("Model Evaluation")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[52]:


joblib.dump(pipeline, 'churn_pipeline.sav')


# In[53]:


X_train_ann = X_train.copy()
X_test_ann = X_test.copy()


# In[54]:


scaler_ann = MinMaxScaler()
X_train_ann[cols_to_scale] = scaler_ann.fit_transform(X_train_ann[cols_to_scale])
X_test_ann[cols_to_scale] = scaler_ann.transform(X_test_ann[cols_to_scale])


# In[55]:


model_ann = Sequential()
model_ann.add(Dense(64, input_dim=X_train_ann.shape[1], activation='relu'))
model_ann.add(Dropout(0.3))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dropout(0.3))
model_ann.add(Dense(1, activation='sigmoid'))


# In[56]:


model_ann.summary()


# In[57]:


model_ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[58]:


model_ann.build(input_shape=(None, X_train_ann.shape[1]))


# In[59]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


# In[60]:


history = model_ann.fit(X_train_ann, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)


# In[61]:


loss, accuracy = model_ann.evaluate(X_test_ann, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")


# In[62]:


y_pred_ann = (model_ann.predict(X_test_ann) > 0.5).astype("int32")


# In[63]:


print(confusion_matrix(y_test, y_pred_ann))


# In[64]:


print(classification_report(y_test, y_pred_ann))


# In[65]:


model_ann.save('ann_churn_model.h5')

