#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import numpy as np


# In[3]:


import pandas as pd


# In[4]:


from sklearn.svm import SVC


# In[5]:


import plotly.express as px


# In[6]:


import statsmodels.api as sm


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


from scipy.stats import loguniform


# In[9]:


import statsmodels.formula.api as smf


# In[10]:


from sklearn.naive_bayes import GaussianNB


# In[11]:


from sklearn.feature_selection import RFECV


# In[ ]:


from tensorflow.keras.optimizers import Adam


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


from tensorflow.keras.models import Sequential


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from tensorflow.keras.layers import Dense, Dropout


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


data_df = pd.read_csv("churn.csv")


# In[ ]:


def dataoveriew(df, message):
    print(f'{message}:\n')
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())


# In[ ]:


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

# In[ ]:


target_instance = data_df["Churn"].value_counts().reset_index()


# In[ ]:


target_instance.columns = ['Category', 'Count']


# In[ ]:


fig = px.pie(
    target_instance,
    values='Count',
    names='Category',
    color='Category',
    color_discrete_sequence=["#FFFF99", "#FFF44F"],  # canary, lemon
    color_discrete_map={"No": "#E30B5C", "Yes": "#50C878"},  # raspberry, emerald
    title='Distribution of Churn'
)


# In[ ]:


fig.show()


# We are trying to predict users that left the company in the previous month. It is a binary classification problem with an unbalance target.
# - Churn: No - 73.5%
# - Churn: Yes - 26.5%

# ### Explore Categorical features

# In[ ]:


def bar(feature, df=data_df ):

    temp_df = df.groupby([feature, 'Churn']).size().reset_index()
    temp_df = temp_df.rename(columns={0:'Count'})

    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100,1) for element in div_list]

    def num_format(list_instance):
        formatted_str = ''
        for index,num in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{num}%, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{num}% & '
            else:
                formatted_str=formatted_str+f'{num}%'
        return formatted_str

    def str_format(list_instance):
        formatted_str = ''
        for index, cat in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{cat}, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{cat} & '
            else:
                formatted_str=formatted_str+f'{cat}'
        return formatted_str


    num_str = num_format(percentage)
    cat_str = str_format(categories)


    churn_colors = ["#FFFF99", "#FFF44F"]
    gender_colors = {"Female": "#E30B5C", "Male": "#50C878"}

    if feature.lower() == "gender":
        fig = px.bar(
            temp_df,
            x=feature,
            y='Count',
            color=feature,
            title=f'Churn rate by {feature}',
            barmode="group",
            color_discrete_map=gender_colors
        )
    else:
        fig = px.bar(
            temp_df,
            x=feature,
            y='Count',
            color='Churn',
            title=f'Churn rate by {feature}',
            barmode="group",
            color_discrete_sequence=churn_colors
        )

    fig.add_annotation(
        text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=1.4,
        y=1.3,
        bordercolor='black',
        borderwidth=1)
    fig.update_layout(
        margin=dict(r=400),
    )
    return fig.show()


# In[ ]:


bar('gender')


# In[ ]:


data_df.loc[data_df.SeniorCitizen==0,'SeniorCitizen'] = "No"


# In[ ]:


data_df.loc[data_df.SeniorCitizen==1,'SeniorCitizen'] = "Yes"


# In[ ]:


bar('SeniorCitizen')


# In[ ]:


bar('Partner')


# In[ ]:


bar('Dependents')


# ***
# **Demographic analysis Insight**: 
# Gender and partner are even distributed with approximate percentage values. The difference in churn is slightly higher in females but the diffreence is negligible. There is a higher proportion of churn amongst younger customers (where SeniorCitizen is No), customers with no partners and customers with no dependents. These analysis on demographic section of data highlights on-senior citizens with no partners and dependents describe a particular segment of customers that are likely to churn.
# ***

# In[ ]:


bar('PhoneService')


# In[ ]:


bar('MultipleLines')


# In[ ]:


bar('InternetService')


# In[ ]:


bar('OnlineSecurity')


# In[ ]:


bar('OnlineBackup')


# In[ ]:


bar('DeviceProtection')


# In[ ]:


bar('TechSupport')


# In[ ]:


bar('StreamingTV')


# In[ ]:


bar('StreamingMovies')


# ***
# **Services that each customer has signed up for Insight**:
# These category of features shows significant variations across their values. If a customer does not have a phone service, he/she cannot have multiple lines. About 90.3% of the customers have phone services and have the higher rate to churn. Customers who have Fibre optic as internet service are more likely to churn, this can happen due to high prices, competition, customer service, and many other reasons. Fiber optic service is much more expensive than DSL which may be one of the reasons why customers churn. Customers with  OnlineSecurity ,OnlineBackup ,DeviceProtection and TechSupport  are more unlikely to churn. Streaming service is not predictive for churn as it evenly distributed to yes and no options.
# ***

# In[ ]:


bar('Contract')


# In[ ]:


bar('PaperlessBilling')


# In[ ]:


bar('PaymentMethod')


# **Payment**:
# ***
# The shorter the contract the higher churn rate as those with longer plans face additional barriers when cancelling prematurely. This clearly explains the motivation for companies to have long-term relationship with their customers. Churn Rate is higher for the customers who opted for paperless billing, About 59.2% of the customers make paperless billing. Customers who pay with electronic check are more likely to churn and this kind of payment is more common than other payment types.
# ***

# ### Explore Numeric features

# In[ ]:


data_df.dtypes


# In[ ]:


try:
    data_df['TotalCharges'] = data_df['TotalCharges'].astype(float)
except ValueError as ve:
    print (ve)


# In[ ]:


data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')


# In[ ]:


data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())


# In[ ]:


def hist(feature):
    group_df = data_df.groupby([feature, 'Churn']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Churn', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["green", "red"])
    fig.show()


# In[ ]:


hist('tenure')


# In[ ]:


hist('MonthlyCharges')


# In[ ]:


hist('TotalCharges')


# ***
# **Customer account information**: The tenure histogram is rightly skewed and shows that majority of customers has been with the telecom company for just the first few months (0-9 months) and the highest rate of churn is also in that first few months (0-9months). 75% of customers who end up leaving Telcom company  do so within their first 30 months. The monthly charge histogram shows that clients with higher monthly charges have a higher churn rate (This suggests that discounts and promotions can be an enticing reason for customers to stay). The total charge trend is quite depict due to variation in frequency.
# Lets bin the numeric features into 3 sections based on quantiles (low, medium and high to get more information from it).
# ***

# In[ ]:


bin_df = pd.DataFrame()


# In[ ]:


bin_df['tenure_bins'] =  pd.qcut(data_df['tenure'], q=3, labels= ['low', 'medium', 'high'])


# In[ ]:


bin_df['MonthlyCharges_bins'] =  pd.qcut(data_df['MonthlyCharges'], q=3, labels= ['low', 'medium', 'high'])


# In[ ]:


bin_df['TotalCharges_bins'] =  pd.qcut(data_df['TotalCharges'], q=3, labels= ['low', 'medium', 'high'])


# In[ ]:


bin_df['Churn'] = data_df['Churn']


# In[ ]:


bar('tenure_bins', bin_df)


# In[ ]:


bar('MonthlyCharges_bins', bin_df)


# In[ ]:


bar('TotalCharges_bins', bin_df)


# ***
# Based on binning, the low tenure and high monthly charge bins have higher churn rates as supported with the previous analysis. While the low Total charge bin has a higher churn rate. 
# ***

# ### Data preprocessing

# In[ ]:


data_df.drop(["customerID"],axis=1,inplace = True)


# In[ ]:


def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})


# In[ ]:


data_df['Churn'] = data_df[['Churn']].apply(binary_map)


# In[ ]:


data_df['gender'] = data_df['gender'].map({'Male':1, 'Female':0})


# In[ ]:


binary_list = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']


# In[ ]:


data_df[binary_list] = data_df[binary_list].apply(binary_map)


# In[ ]:


data_df = pd.get_dummies(data_df, drop_first=True)


# In[ ]:


corr = data_df.corr()


# In[ ]:


fig = px.imshow(corr,width=1000, height=1000)


# In[ ]:


fig.show()


# Correlation is a statistical term is a measure on linear relationship with two variables. Features with high correlation are more linearly dependent and have almost the same effect on the dependent variable. So when two features have a high correlation, we can drop one of the two features.

# In[ ]:


all_columns = [column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_") for column in data_df.columns]


# In[ ]:


data_df.columns = all_columns


# In[ ]:


glm_columns = [e for e in all_columns if e not in ['customerID', 'Churn']]


# In[ ]:


glm_columns = ' + '.join(map(str, glm_columns))


# In[ ]:


glm_model = smf.glm(formula=f'Churn ~ {glm_columns}', data=data_df, family=sm.families.Binomial())


# In[ ]:


res = glm_model.fit()


# In[ ]:


print(res.summary())


# In[ ]:


np.exp(res.params)


# In[ ]:


sc = MinMaxScaler()


# In[ ]:


data_df['tenure'] = sc.fit_transform(data_df[['tenure']])


# In[ ]:


data_df['MonthlyCharges'] = sc.fit_transform(data_df[['MonthlyCharges']])


# In[ ]:


data_df['TotalCharges'] = sc.fit_transform(data_df[['TotalCharges']])


# #### Creating a baseline model

# In[ ]:


X = data_df.drop('Churn', axis=1)


# In[ ]:


y = data_df['Churn']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)  


# In[ ]:


def modeling(alg, alg_name, params={}):
    model = alg(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ",acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ",pre_score)
        rec_score = recall_score(y_true, y_pred)                            
        print("recall: ",rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ",f_score)

    print_scores(alg, y_test, y_pred)
    return model


# In[ ]:


log_model = modeling(LogisticRegression, 'Logistic Regression')


# In[ ]:


log = LogisticRegression()


# In[ ]:


rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")


# In[ ]:


rfecv.fit(X, y)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
    rfecv.cv_results_['mean_test_score']
)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()

print("The optimal number of features: {}".format(rfecv.n_features_))


# In[ ]:


X_rfe = X.iloc[:, rfecv.support_]

print("\"X\" dimension: {}".format(X.shape))
print("\"X\" column list:", X.columns.tolist())
print("\"X_rfe\" dimension: {}".format(X_rfe.shape))
print("\"X_rfe\" column list:", X_rfe.columns.tolist())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.3, random_state=50)  


# In[ ]:


log_model = modeling(LogisticRegression, 'Logistic Regression Classification')


# In[ ]:


svc_model = modeling(SVC, 'SVC Classification')


# In[ ]:


rf_model = modeling(RandomForestClassifier, "Random Forest Classification")


# In[ ]:


dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classification")


# In[ ]:


nb_model = modeling(GaussianNB, "Naive Bayes Classification")


# In[ ]:


model = LogisticRegression()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 1000)
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
result = search.fit(X_rfe, y)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# In[ ]:


params = result.best_params_
params


# In[ ]:


log_model = modeling(LogisticRegression, 'Logistic Regression Classification', params=params)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.3, random_state=50)


# In[ ]:


model_ann = Sequential()


# In[ ]:


model_ann.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))


# In[ ]:


model_ann.add(Dropout(0.3))


# In[ ]:


model_ann.add(Dense(32, activation='relu'))


# In[ ]:


model_ann.add(Dropout(0.3))


# In[ ]:


model_ann.add(Dense(1, activation='sigmoid'))


# In[ ]:


model_ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


# In[ ]:


history = model_ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])


# In[ ]:


loss, accuracy = model_ann.evaluate(X_test, y_test)


# In[ ]:


print(f"Test Accuracy: {accuracy:.4f}")


# In[ ]:


y_pred = (model_ann.predict(X_test) > 0.5).astype("int32")


# In[ ]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


filename = 'model.sav'
joblib.dump(log_model, filename)

